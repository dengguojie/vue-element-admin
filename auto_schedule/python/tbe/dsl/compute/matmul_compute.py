#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
mmad_compute
"""
from __future__ import absolute_import

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.utils.errormgr import error_manager_cube
from tbe.dsl.base.operation import in_dynamic
from tbe.dsl.compute.util import int_ceil_div
from tbe.dsl.compute.util import get_value
from tbe.dsl.compute.util import shape_to_list


BATCH_MATMUL_LENGTH = 5
DTYPE_TRANS_MAP = {
    "int4": "S4",
    "int8": "B8",
    "float16": "F16",
    "float32": "F32",
    "int32": "S32",
    "bfloat16": "BF16"
}


class MatMulComputeParam:
    """
    be used by gemm_tilingcase
    """
    tiling_info_dict = {}
    dynamic_mode = None
    batch_a = False
    batch_b = False
    format_a = "Fractal_NZ"
    format_b = "Fractal_NZ"
    m_var_name = None
    k_var_name = None
    n_var_name = None
    block_in = tbe_platform.BLOCK_IN
    block_out = tbe_platform.BLOCK_OUT
    block_reduce = tbe_platform.BLOCK_REDUCE

    def __init__(self) -> None:
        pass


class MatMulCompute:
    """
    algorithm: mmad
    calculating matrix multiplication, C=A*B+bias

    Parameters:
    tensor_a : the first tensor a

    tensor_b : second tensor b with the same type and shape with a

              If tensor_a/tensor_b is int8/uint8,then L0A must be 16*32,L0B
              must be 32*16.
              If A is transpose , then AShape classification matrix must be
              32*16 in gm/L1,then it is 16*32 in L0A.
              If B is transpose , then BShape classification matrix must be
              16*32 in gm/L1,then it is 32*16 in L0B.

    trans_a : if True, a needs to be transposed

    trans_b : if True, b needs to be transposed

    format_a: the format of tensor a, support FRACTAL_NZ, ND
              default is "ND"

    format_b: the format of tensor b, support FRACTAL_NZ, ND
              default is "ND"

    dst_dtype: output data type, support "float32", default is "float32"

    tensor_bias :the bias with used to add

    format_out: output format, now support ND,Nz

    kernel_name: kernel name, default is "MatMul"

    op_type: op_type

    Returns None
    """
    def __init__(self, tensor_a,
        tensor_b, trans_a=False, trans_b=False, format_a="ND", format_b="ND",
        dst_dtype="float32", tensor_bias=None, format_out=None, kernel_name="MatMul", op_type=""):
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.format_a = format_a
        self.format_b = format_b
        self.tensor_bias = tensor_bias
        self.src_dtype = tensor_a.dtype
        self.dst_dtype = dst_dtype
        self.kernel_name = kernel_name
        self.block_in = tbe_platform.BLOCK_IN
        self.block_out = tbe_platform.BLOCK_OUT
        self.block_reduce = tbe_platform.CUBE_MKN[self.src_dtype]["mac"][1]
        self.origin_reduce_axis = 0
        self.matrix_type = "int32" if self.src_dtype == "int8" else "float32"
        self.origin_m_shape = 0
        self.origin_n_shape = 0
        self.op_type = op_type
        self.m_shape = 0
        self.km_shape = 0
        self.n_shape = 0
        self.format_out = format_out

        # the parameters to be calculated
        self.c_matrix = None
        self.batch_shape_a = None
        self.batch_shape_b = None
        self.kn_shape = None

    @staticmethod
    def get_trans_flag(transpose_a, transpose_b):
        """
        get trans flag inorder to get tiling
        """
        trans_flag = 1
        if transpose_a:
            if transpose_b:
                trans_flag = 4
            else:
                trans_flag = 2
        elif transpose_b:
            trans_flag = 3
        return trans_flag

    @staticmethod
    def get_a_shape_in_nc1hwc0(tensor_a_l0a):
        """
        get a shape's format nc1hwc0 inorder to get tiling
        """
        if MatMulComputeParam.batch_a:
            a_5hd_shape = [tensor_a_l0a.shape[0], tensor_a_l0a.shape[2], tensor_a_l0a.shape[1], 16, 16]
        else:
            a_5hd_shape = [1, tensor_a_l0a.shape[1], tensor_a_l0a.shape[0], 16, 16]
        return a_5hd_shape

    @staticmethod
    def get_b_shape_in_nc1hwc0(tensor_b_l0b):
        """
        get b shape's format nc1hwc0 inorder to get tiling
        """
        return  [tensor_b_l0b.shape[-4] * 16, tensor_b_l0b.shape[-3], 1, 1, 16]

    @staticmethod
    def _get_mad_b_indices(indices, reduce_axis):
        """
        get b_matrix indices depend on batch
        """
        reduce_kb = reduce_axis[0]
        reduce_kp = reduce_axis[1]
        if MatMulComputeParam.batch_b:
            b_indices = [*indices[:-4], reduce_kb, indices[-4], indices[-1], reduce_kp]
        else:
            b_indices = [reduce_kb, indices[-4], indices[-1], reduce_kp]
        return b_indices

    def compute_matmul(self):
        """
        MatMul enter

        Input None
        return result in self.c_matrix
        ---------------------------------
        Return None
        """
        # set l1 shape
        if not in_dynamic():
            self._check_attrs()
        self._get_l1_shape()

        tensor_b_length = len(self.tensor_b.shape)
        MatMulComputeParam.batch_a = self._check_batch_a()
        MatMulComputeParam.batch_b = (tensor_b_length == BATCH_MATMUL_LENGTH)

        a_matrix = self._get_a_matrix()
        b_matrix = self._get_b_matrix()
        self.c_matrix = self._compute_c_matrix(a_matrix, b_matrix)
        self._set_dynamic_param(a_matrix, b_matrix)

    def _handle_front_trans_fusion(self):
        """
        for 5hd input, trans 5hd to fractal_z
        """
        src_n, src_c1, src_h, src_w, src_c0 = tuple(i.value for i in self.tensor_a.shape)
        dst_n1 = int_ceil_div(src_n, self.block_in)
        self.origin_reduce_axis = src_c1 * src_h * src_w * src_c0
        fz_shape = (
            src_c1 * src_h * src_w,
            dst_n1,
            self.block_in,
            src_c0
        )
        self.tensor_a = tvm.compute(
            fz_shape,
            lambda * indices: self.tensor_a(
                indices[-3] * self.block_in + indices[-2],
                indices[-4] // (src_h * src_w),
                indices[-4] // src_w % src_h,
                indices[-4] % src_w,
                indices[-1]
            ),
            name=self.tensor_a.name + "_fractal_z",
            tag="5HD_trans_FZ"
        )
        self.m_shape = get_value(self.tensor_a.shape[-3])
        self.km_shape = get_value(self.tensor_a.shape[-4])
        self.origin_m_shape = src_n

    def _al1_trans_nd2nz(self):
        """
        for fully_connection ND input, trans ND to FRACTAL_NZ
        """
        nd_shape = tuple(i.value for i in self.tensor_a.shape)
        nz_shape = (
            int_ceil_div(nd_shape[-1], self.block_reduce),
            int_ceil_div(nd_shape[-2], self.block_in),
            self.block_in,
            self.block_reduce
        )
        d_axis_origin_length = nd_shape[-1]
        self.tensor_a = tvm.compute(
            nz_shape,
            lambda *indices: tvm.select(
                tvm.all((indices[-4] * self.block_reduce + indices[-1]) < d_axis_origin_length),
                self.tensor_a(*indices[:-4],
                              indices[-3] * self.block_in + indices[-2],
                              indices[-4] * self.block_reduce + indices[-1])
            ),
            name=self.tensor_a.name + "_fractal",
            attrs={"ori_format": "ND", "ori_shape": nd_shape, "format": "FRACTAL_NZ"},
            tag="ND_trans_NZ"
        )
        self.m_shape = get_value(self.tensor_a.shape[-3]) if self.trans_a else get_value(self.tensor_a.shape[-4])
        self.km_shape = get_value(self.tensor_a.shape[-4]) if self.trans_a else get_value(self.tensor_a.shape[-3])

    def _get_a_matrix_fp32(self, temp_tensor_a):
        """get a_matrix for float32 input

        Parameters
        ----------
        temp_tensor_a : tensor

        Returns
        -------
        tensor
        """
        block_reduce_multiple_in = self.block_in // self.block_reduce
        if not self.trans_a:
            a_matrix_shape = [
                int_ceil_div(self.m_shape, block_reduce_multiple_in),
                self.km_shape * block_reduce_multiple_in,
                self.block_in,
                self.block_reduce]
        else:
            a_matrix_shape = [
                self.m_shape,
                self.km_shape,
                self.block_in,
                self.block_reduce]
        if self.batch_shape_a:
            a_matrix_shape.insert(0, self.batch_shape_a)
        if not self.trans_a:
            a_matrix = tvm.compute(
                a_matrix_shape,
                lambda *indices:
                    temp_tensor_a(*indices[:-4],
                                  indices[-4] * block_reduce_multiple_in + indices[-2] // self.block_reduce,
                                  (indices[-3] * self.block_reduce + indices[-1]) // self.block_in,
                                  (indices[-3] * self.block_reduce + indices[-1]) % self.block_in,
                                  indices[-2] % self.block_reduce),
                name="tensor_a_matrix",
                attrs={"transpose_a": "false"}
                )
        else:
            a_matrix = tvm.compute(
                a_matrix_shape,
                lambda *indices: temp_tensor_a(*indices[:-4], indices[-3],
                                               indices[-4], *indices[-2:]),
                name="tensor_a_matrix",
                attrs={"transpose_a": "true"}
                )
        return a_matrix

    def _get_a_matrix(self):
        """ compute matrix for mad
        Input : None
        support func:
            fp16 input:
                Nz->Zz
        ---------------------------------
        Return : tensor, Zz matrix for mad
        """
        if self.format_a == "ND":
            if "NHWC_trans_5HD" in self.tensor_a.op.tag:
                self._handle_front_trans_fusion()
            else:
                self._al1_trans_nd2nz()
        if self.src_dtype == "int8" and not self.trans_a:
            block_reduce_multiple_in = self.block_reduce // self.block_in
            a_matrix_shape = [self.m_shape * block_reduce_multiple_in,
                              int_ceil_div(self.km_shape, block_reduce_multiple_in),
                              self.block_in,
                              self.block_reduce]
            if MatMulComputeParam.batch_a:
                a_matrix_shape.insert(0, self.batch_shape_a)
            a_matrix = tvm.compute(
                a_matrix_shape, lambda *indices:
                self.tensor_a(*indices[:-4],
                              indices[-4] // block_reduce_multiple_in,
                              indices[-3] * block_reduce_multiple_in + indices[-1] // self.block_in,
                              indices[-1] % self.block_in,
                              indices[-2] + indices[-4] % block_reduce_multiple_in * self.block_in),
                name="tensor_a_matrix",
                attrs={"transpose_a": "false"})
        elif self.src_dtype == "float32":
            a_matrix = self._get_a_matrix_fp32(self.tensor_a)
        else:
            a_matrix_shape = [self.m_shape, self.km_shape, self.block_in, self.block_reduce]
            if MatMulComputeParam.batch_a:
                a_matrix_shape.insert(0, self.batch_shape_a)
            if self.trans_a:
                a_matrix = tvm.compute(
                    a_matrix_shape,
                    lambda *indices: self.tensor_a(*indices[:-4], indices[-3], indices[-4], *indices[-2:]),
                    name="tensor_a_matrix",
                    attrs={"transpose_a": "true"})
            else:
                a_matrix = tvm.compute(
                    a_matrix_shape,
                    lambda *indices: self.tensor_a(*indices[:-2], indices[-1], indices[-2]),
                    name="tensor_a_matrix",
                    attrs={"transpose_a": "false"})

        return a_matrix

    def _get_b_matrix_fp32(self, temp_tensor_b):
        """get b_matrix for float32 input

        Parameters
        ----------
        temp_tensor_b : tensor

        Returns
        -------
        tensor
        """
        block_reduce_multiple_in = self.block_in // self.block_reduce
        if not self.trans_b:
            b_matrix_shape = [
                self.kn_shape,
                self.n_shape,
                self.block_out,
                self.block_reduce]
        else:
            b_matrix_shape = [
                self.kn_shape * block_reduce_multiple_in,
                int_ceil_div(self.n_shape, block_reduce_multiple_in),
                self.block_out,
                self.block_reduce]
        if self.batch_shape_b:
            b_matrix_shape.insert(0, self.batch_shape_b)
        if not self.trans_b:
            b_matrix = tvm.compute(
                b_matrix_shape,
                lambda *indices: temp_tensor_b(*indices),
                name="tensor_b_matrix",
                attrs={"transpose_b": "false"}
            )
        else:
            b_matrix = tvm.compute(
                b_matrix_shape,
                lambda *indices:
                    temp_tensor_b(*indices[:-4],
                                  indices[-3] * block_reduce_multiple_in + indices[-2] // self.block_reduce,
                                  (indices[-4] * 8 + indices[-1]) // 16,
                                  (indices[-4] * 8 + indices[-1]) % 16,
                                  indices[-2] % 8),
                name="tensor_b_matrix",
                attrs={"transpose_b": "true"}
            )
        return b_matrix

    def _get_b_matrix(self):
        """ compute matrix for mad
        Input : None
        support func:
            fp16 input:
                Nz->Zn
                ND->Nz->Zn
        ---------------------------------
        Return : tensor, Zn matrix for mad
        """
        # to Zn
        if self.src_dtype == "float32":
            b_matrix = self._get_b_matrix_fp32(self.tensor_b)
        else:
            if self.trans_b:
                if self.src_dtype == "int8":
                    block_reduce_multiple_out = self.block_reduce // self.block_in
                    b_matrix_shape = [int_ceil_div(self.kn_shape, block_reduce_multiple_out),
                                      self.n_shape * block_reduce_multiple_out,
                                      self.block_out,
                                      self.block_reduce]
                    if MatMulComputeParam.batch_b:
                        b_matrix_shape.insert(0, self.batch_shape_b)
                    b_matrix = tvm.compute(
                        b_matrix_shape,
                        lambda *indices:
                        self.tensor_b(*indices[:-4],
                                      indices[-3] // block_reduce_multiple_out,
                                      indices[-4] * block_reduce_multiple_out + indices[-1] // self.block_out,
                                      indices[-1] % self.block_out,
                                      indices[-2] + indices[-3] % block_reduce_multiple_out * self.block_out),
                                      name="tensor_b_matrix",
                                      attrs={"transpose_b":"true"})
                else:
                    b_matrix_shape = [self.kn_shape, self.n_shape, self.block_out, self.block_reduce]
                    if MatMulComputeParam.batch_b:
                        b_matrix_shape.insert(0, self.batch_shape_b)
                    b_matrix = tvm.compute(
                        b_matrix_shape,
                        lambda *indices: self.tensor_b(*indices[:-4], indices[-3],
                                                       indices[-4], indices[-1], indices[-2]),
                        name="tensor_b_matrix",
                        attrs={"transpose_b": "true"})
            else:
                b_matrix = tvm.compute(
                    self.tensor_b.shape,
                    lambda *indices: self.tensor_b(*indices),
                    name="tensor_b_matrix",
                    attrs={"transpose_b": "false"}
                )

        return b_matrix

    def _get_pre_conv_mode(self, tensor_c_matrix):
        """get dtype trans mode
        Input:
            tensor_c_matrix: tensor, c_matrix on l0c
        ---------------------------------
        Return:
            str, dtype trans mode
        """
        conv_mode = DTYPE_TRANS_MAP.get(tensor_c_matrix.dtype) + "2" + DTYPE_TRANS_MAP.get(self.dst_dtype)
        return conv_mode

    def _handle_5hd_output(self, l0c_shape, tensor_c_matrix):
        """handle 5hd output, nz2nd must be implemented by fixpipe op
        Input:
            l0c_shape: list, shape_of l0c
            tensor_c_matrix: tensor, c_matrix on l0c
        ---------------------------------
        Return:
            tensor, nd tensor of fc
        """
        out_shape = (self.origin_m_shape, self.origin_n_shape)
        op_dict = {
            "post_transform": "NZ2ND",
            "pre_conv": self._get_pre_conv_mode(tensor_c_matrix)
        }
        attrs = {
            "vector_params": [],
            "vector_tensors": [],
            "nz2nd_flag": True,
            "anti_quant_flag": False
        }
        fixpipe_tensor = tvm.compute(
            l0c_shape,
            lambda *indices: tvm.fixpipe_op(
                tensor_c_matrix(*indices),
                self.dst_dtype,
                op_dict=op_dict),
            name="fixpipe",
            tag="fixpipe",
            attrs=attrs)
        tensor_c_gm = tvm.compute(
            out_shape,
            lambda *indices: fixpipe_tensor(*indices[:-2],
                                            indices[-1] // self.block_in,
                                            indices[-2] // self.block_out,
                                            indices[-2] % self.block_out,
                                            indices[-1] % self.block_in),
            name="tensor_c_gm",
            tag="gemm"
        )
        return tensor_c_gm

    def _compute_c_gm(self, tensor_c_matrix, output_shape):
        """
        compute c_gm from l0c matrix-
        """
        l0c_shape = shape_to_list(tensor_c_matrix.shape)
        ori_shape = [self.origin_m_shape, self.origin_n_shape]
        if MatMulComputeParam.batch_a:
            ori_shape.insert(0, self.batch_shape_a)
        if self.format_out == "NC1HWC0":
            # ND output
            tensor_c_gm = self._handle_5hd_output(l0c_shape, tensor_c_matrix)
        elif self.dst_dtype == "float32" and self.src_dtype == "float32":
            output_shape[-4] = int_ceil_div(self.origin_n_shape, self.block_reduce)
            output_shape[-1] = self.block_reduce
            tensor_c_gm = tvm.compute(
                output_shape,
                lambda *indices: tensor_c_matrix(*indices[:-4],
                                                 (indices[-4] * self.block_reduce + indices[-1]) // self.block_out,
                                                 indices[-3],
                                                 indices[-2],
                                                 (indices[-4] * self.block_reduce + indices[-1]) %
                                                 self.block_out).astype(self.dst_dtype),
                tag="gemm",
                name="tensor_c_gm",
                attrs={"ori_shape": ori_shape,
                       "shape": output_shape})
        else:
            tensor_c_gm = tvm.compute(output_shape,
                                    lambda *indices: tensor_c_matrix(*indices).astype(self.dst_dtype),
                                    tag="gemm",
                                    name="tensor_c_gm",
                                    attrs={"ori_shape": ori_shape,
                                           "shape": output_shape})
        return tensor_c_gm


    def _compute_c_matrix(self, a_matrix_in, b_matrix_in):
        """ MatMul calculation
        Input:
            a_matrix_in: tensor, a_matrix in l0a
            b_matrix_in: tensor, b_matrix in l0b
        support func:
            MatMul, Nz->ND
        ---------------------------------
        Return:
            tensor, MatMul result
        """
        reduce_kb = tvm.reduce_axis((0, a_matrix_in.shape[-3]), name="kb")
        reduce_kp = tvm.reduce_axis((0, self.block_reduce), name="kp")
        l0c_shape = [b_matrix_in.shape[-3], a_matrix_in.shape[-4], self.block_in, self.block_out]
        output_shape = [int_ceil_div(self.origin_n_shape, self.block_out),
                        int_ceil_div(self.origin_m_shape, self.block_in),
                        self.block_in, self.block_out]
        if MatMulComputeParam.batch_a:
            l0c_shape.insert(0, self.batch_shape_a)
            output_shape.insert(0, self.batch_shape_a)
        if self.tensor_bias is None:
            tensor_c_matrix = tvm.compute(
                l0c_shape,
                lambda *indices: tvm.sum(tvm.select(
                    tvm.all(reduce_kb.var * self.block_reduce + reduce_kp.var < self.origin_reduce_axis),
                    (a_matrix_in(*indices[:-4], indices[-3], reduce_kb, indices[-2], reduce_kp) *
                     b_matrix_in(*self._get_mad_b_indices(indices, [reduce_kb, reduce_kp]))
                     ).astype(self.matrix_type)),
                    axis=[reduce_kb, reduce_kp]),
                name="tensor_c_matrix",
                tag="gemm",
                attrs={"kernel_name": self.kernel_name})
        else:
            tensor_c_matrix = tvm.compute(
                l0c_shape,
                lambda *indices: tvm.sum(tvm.select(
                    tvm.all(reduce_kb.var * self.block_reduce + reduce_kp.var < self.origin_reduce_axis),
                    (a_matrix_in(*indices[:-4], indices[-3], reduce_kb, indices[-2], reduce_kp) *
                     b_matrix_in(*self._get_mad_b_indices(indices, [reduce_kb, reduce_kp]))
                     ).astype(self.matrix_type) +
                    self.tensor_bias[indices[-4]*self.block_out + indices[-1]]),
                    axis=[reduce_kb, reduce_kp]),
                name='tensor_c_matrix',
                tag="gemm",
                attrs={"kernel_name": self.kernel_name})
        tensor_c_gm = self._compute_c_gm(tensor_c_matrix, output_shape)
        return tensor_c_gm

    def _check_attrs(self):
        if "ori_shape" not in self.tensor_a.op.attrs:
            error_manager_cube.raise_err_specific("MatMul", "tensor_a must have attr ori_shape")
        if "ori_shape" not in self.tensor_b.op.attrs:
            error_manager_cube.raise_err_specific("MatMul", "tensor_b must have attr ori_shape")

    def _get_fully_connection_n_shape(self):
        """get origin n shape for fully-connection
        1) NC1HWC0 * (Cin1HW Co1 Co0 Cin0) = NC1HWC0, n shape is Co1*Co0 in Fz shape
        2) NC1HWC0/FRACTAL_NZ * (Cin1 Co1 Co0 Cin0) = FRACTAL_NZ, n shape is N dim origin_shape_b of NCHW/NHWC format
        """
        if self.format_out == "NC1HWC0":
            self.origin_n_shape = get_value(self.tensor_b.shape[-3]) * get_value(self.tensor_b.shape[-2])
        else:
            # FRACTAL_NZ output
            if "ori_format" in self.tensor_b.op.attrs:
                ori_format_b = self.tensor_b.op.attrs["ori_format"].value
            else:
                ori_format_b = "NCHW"
            n_index = ori_format_b.find('N')
            if n_index < 0:
                error_manager_cube.raise_err_specific("FullyConnection", "origin format of input2 is illegal")
            origin_shape_b = self.tensor_b.op.attrs["ori_shape"]
            self.origin_n_shape = origin_shape_b[n_index]

    def _get_al1_shape(self):
        if self._check_batch_a():
            self.batch_shape_a = get_value(self.tensor_a.shape[0])

        if self.format_a == "FRACTAL_NZ":
            # trans shape -> batch, K, M, 16, 16/32
            # not trans shape -> batch, M, K, 16, 16/32
            self.m_shape = get_value(self.tensor_a.shape[-3]) if self.trans_a else get_value(self.tensor_a.shape[-4])
            self.km_shape = get_value(self.tensor_a.shape[-4]) if self.trans_a else get_value(self.tensor_a.shape[-3])
            if in_dynamic():
                self.origin_m_shape = self.m_shape * get_value(self.tensor_a.shape[-2])
                self.origin_reduce_axis = self.km_shape * get_value(self.tensor_a.shape[-1])
            else:
                origin_shape = self.tensor_a.op.attrs["ori_shape"]
                self.origin_m_shape = origin_shape[-2] if self.trans_a else origin_shape[-1]
                self.origin_reduce_axis = origin_shape[-1] if self.trans_a else origin_shape[-2]
        elif self.format_a == "ND":
            nd_shape = self.tensor_a.shape
            self.trans_a = not self.trans_a
            self.origin_m_shape = get_value(nd_shape[-2]) if self.trans_a else get_value(nd_shape[-1])
            self.origin_reduce_axis = get_value(nd_shape[-1]) if self.trans_a else get_value(nd_shape[-2])
        else:
            error_manager_cube.raise_err_specific("MatMul", "tensor_a only supported NZ or ND format")

    def _get_bl1_shape(self):
        if len(self.tensor_b.shape) == BATCH_MATMUL_LENGTH:
            self.batch_shape_b = get_value(self.tensor_b.shape[0])

        if self.format_b in ("FRACTAL_NZ", "FRACTAL_Z"):
            # trans shape -> batch, N, K, 16, 16/32
            # not trans shape -> batch, K, N, 16, 16/32
            self.kn_shape = get_value(self.tensor_b.shape[-3]) if self.trans_b else get_value(self.tensor_b.shape[-4])
            self.n_shape = get_value(self.tensor_b.shape[-4]) if self.trans_b else get_value(self.tensor_b.shape[-3])
            if in_dynamic():
                self.origin_n_shape = self.n_shape * get_value(self.tensor_b.shape[-2])
            elif self.op_type == "FullyConnection":
                self._get_fully_connection_n_shape()
            else:
                origin_shape = self.tensor_b.op.attrs["ori_shape"]
                if self.format_b == "FRACTAL_NZ":
                    self.origin_n_shape = origin_shape[-1] if self.trans_b else origin_shape[-2]
                else:
                    self.origin_n_shape = origin_shape[-2] if self.trans_b else origin_shape[-1]
        else:
            error_manager_cube.raise_err_specific("MatMul", "tensor_b only supported NZ and Z format")

    def _get_l1_shape(self):
        """ get shape about m,k,n
        Input: None
        ---------------------------------
        Return: None
        """
        self._get_al1_shape()
        self._get_bl1_shape()

    def _check_batch_a(self):
        """
        check if tensor_a has batch:
        for nz input, the first dim of 5 dims input is batch
        for 5hd input, have no batch(op_type is fc, nhwc--transdata_compute-->5hd)
        """
        tensor_a_length = len(self.tensor_a.shape)
        return (tensor_a_length == BATCH_MATMUL_LENGTH) and ("NHWC_trans_5HD" not in self.tensor_a.op.tag)

    def _set_dynamic_param(self, a_matrix, b_matrix):
        """
        set MatmulComputeParam to support for tilingcase
        """
        if in_dynamic():
            if MatMulComputeParam.batch_a:
                MatMulComputeParam.dynamic_mode = "dynamic_mknb"
            else:
                MatMulComputeParam.dynamic_mode = "dynamic_mkn"

        MatMulComputeParam.m_var_name = "m"
        MatMulComputeParam.n_var_name = "n"
        MatMulComputeParam.k_var_name = "k"
        MatMulComputeParam.tiling_info_dict = {
            "A_shape": self.get_a_shape_in_nc1hwc0(a_matrix),
            "B_shape": self.get_b_shape_in_nc1hwc0(b_matrix),
            "C_shape": None,
            "A_dtype": self.tensor_a.dtype,
            "B_dtype": self.tensor_b.dtype,
            "C_dtype": self.c_matrix.dtype,
            "mad_dtype": self.matrix_type,
            "padl": 0,
            "padr": 0,
            "padu": 0,
            "padd": 0,
            "strideH": 1,
            "strideW": 1,
            "strideH_expand": 1,
            "strideW_expand": 1,
            "dilationH": self.get_trans_flag(not self.trans_a, not self.trans_b),
            "dilationW": 1,
            "group": 1,
            "fused_double_operand_num": 0,
            "bias_flag": (self.tensor_bias is not None),
            "op_tag": "matmul",
            "op_type": "matmul",
            "kernel_name": self.kernel_name,
            "dynamic_shape_flag": True,
            "trans_a": not self.trans_a,
            "trans_b": not self.trans_b
        }
