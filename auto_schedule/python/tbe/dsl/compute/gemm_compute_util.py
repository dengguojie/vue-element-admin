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
gemm compute util
"""
from functools import reduce as functools_reduce
from tbe.common.utils import broadcast_shapes
from tbe.common.utils import shape_util
from tbe.dsl.base.operation import in_dynamic
from tbe.tvm import api as tvm
from tbe.dsl.compute.util import int_ceil_div


class FormatCompute(object):
    """
    the format compute for gemm
    """
    def __init__(self):
        pass

    @staticmethod
    def p2p_copy(tensor_src, shape_src, tensor_name):
        """
        do point to point copy
        input params:
            tensor_src: the source tensor
            shape_src: the shape of the source tensor
            tensor_name: the name of dst tensor
        return:
            the dst tensor
        """
        tensor_dst = tvm.compute(
            shape_src,
            lambda *indices: tensor_src(*indices),
            name=tensor_name
        )
        return tensor_dst

    @staticmethod
    def compute_nd2zz(ori_tensor, compute_params):
        """
        reshape nd2Zz by normal way
        input params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
           the tensor format Zz
        """
        tensor_name = compute_params.get("tensor_name")
        block_in = compute_params.get("block_in")
        block_reduce = compute_params.get("block_reduce")
        data_flow = compute_params.get("data_flow")
        mode_info = compute_params.get("mode_info", "none")
        trans = compute_params.get("trans")
        int82int32_trans_flag = (data_flow == "int82int32") and trans
        if int82int32_trans_flag:
            return FormatCompute._compute_nd2zz_int8_trans(ori_tensor, compute_params)
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        if trans:
            tensor_matrix_shape = ori_tensor_shape[:-2] + [
                int_ceil_div(ori_tensor_shape[-1], block_in),
                int_ceil_div(ori_tensor_shape[-2], block_reduce),
                block_in,
                block_reduce
            ]
            tensor_l1_shape = tensor_matrix_shape[:-2] + [block_reduce, block_in]
            l1_tensor = tvm.compute(tensor_l1_shape,
                                    lambda *indices: ori_tensor(*indices[:-4],
                                                                indices[-3]*block_reduce + indices[-2],
                                                                indices[-4]*block_in + indices[-1]),
                                    name="tensor_a_l1")
            lambda_expression = lambda *indices: l1_tensor(*indices[:-2], indices[-1], indices[-2])
        else:
            tensor_matrix_shape = ori_tensor_shape[:-2] + [
                int_ceil_div(ori_tensor_shape[-2], block_in),
                int_ceil_div(ori_tensor_shape[-1], block_reduce),
                block_in,
                block_reduce
            ]
            lambda_expression = lambda *indices: ori_tensor(*indices[:-4],
                                                            indices[-4]*block_in + indices[-2],
                                                            indices[-3]*block_reduce + indices[-1])
        res = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info}
        )
        return res

    @staticmethod
    def compute_nd2zz_gevm(ori_tensor, compute_params):
        """
        reshape nd2Zz by normal way and gevm mode
        input params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format Zz
        """
        data_flow = compute_params.get("data_flow")
        mode_info = compute_params.get("mode_info", "none")
        trans = compute_params.get("trans")
        block_in = compute_params.get("block_in")
        block_reduce = compute_params.get("block_reduce")
        int82int32_trans_flag = (data_flow == "int82int32") and trans
        if int82int32_trans_flag:
            return FormatCompute._compute_nd2zz_int8_trans(ori_tensor, compute_params)
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        if trans:
            tensor_matrix_shape = ori_tensor_shape[:-2] + [
                int_ceil_div(ori_tensor_shape[-1], block_in),
                int_ceil_div(ori_tensor_shape[-2], block_reduce),
                block_in, block_reduce
            ]
            lambda_expression = lambda *indices: ori_tensor(*indices[:-4],
                                                            indices[-3]*block_reduce + indices[-1],
                                                            indices[-4]*block_in + indices[-2])
        else:
            tensor_matrix_shape = ori_tensor_shape[:-2] + [
                int_ceil_div(ori_tensor_shape[-2], block_in),
                int_ceil_div(ori_tensor_shape[-1], block_reduce),
                block_in, block_reduce
            ]
            lambda_expression = lambda *indices: ori_tensor(*indices[:-4], indices[-4]*block_in + indices[-2],
                                                            indices[-3]*block_reduce + indices[-1])
        res_fract = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name=compute_params.get("tensor_name"),
            attrs={"mode": mode_info}
        )
        res_matrix = tvm.compute(
            tensor_matrix_shape,
            lambda *indices: res_fract(*indices[:-4], 0, indices[-3], 0, indices[-1]),
            name="tensor_a_zz",
            attrs={"mode": mode_info}
        )
        return res_matrix

    @staticmethod
    def compute_nd2zn(ori_tensor, compute_params):
        """
        reshape nd2Zn by normal way
        input params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format Zn
        """
        tensor_name = compute_params.get("tensor_name")
        block_out = compute_params.get("block_out")
        block_reduce = compute_params.get("block_reduce")
        mode_info = compute_params.get("mode_info")
        trans = compute_params.get("trans")
        data_flow = compute_params.get("data_flow")
        int82int32_no_trans_flag = (data_flow == "int82int32") and (not trans)
        if int82int32_no_trans_flag:
            return FormatCompute._compute_nd2zn_int8_no_trans(ori_tensor, compute_params)

        if trans:
            fract_shape, lambda_expression = FormatCompute._lambda_nd2nz(ori_tensor, block_out, block_reduce)
        else:
            fract_shape, lambda_expression = FormatCompute._lambda_nd2zn(ori_tensor, block_reduce, block_out)

        res = tvm.compute(
            fract_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info}
        )

        return res

    @staticmethod
    def compute_nd2zn_vnchwconv(ori_tensor, compute_params):
        """
        reshape nd2Zn by vnchwconv [k, n] --> [k / block_reduce, n, block_reduce] -- >
        [k / block_reduce, n / block_out, block_out, block_reduce]
        input params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format is FRACTAL_Z
        """
        tensor_name = compute_params.get("tensor_name")
        block_out = compute_params.get("block_out")
        block_reduce = compute_params.get("block_reduce")
        trans = compute_params.get("trans")
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        tensor_fract_shape = ori_tensor_shape[:-2] + [
            ori_tensor_shape[-2] // block_reduce,
            ori_tensor_shape[-1],
            block_reduce
        ]
        tensor_fract = tvm.compute(
            tensor_fract_shape,
            lambda *indices: ori_tensor(*indices[:-3], indices[-3]*block_reduce + indices[-1], indices[-2]),
            name="{}_fract".format(tensor_name)
        )
        if trans:
            tensor_matrix_shape = tensor_fract_shape[:-3] + [
                tensor_fract_shape[-2] // block_out,
                tensor_fract_shape[-3],
                block_reduce,
                block_out
            ]
            lambda_expression = lambda *indices: tensor_fract(*indices[:-4], indices[-3],
                                                              indices[-4]*block_reduce + indices[-1],
                                                              indices[-2])
        else:
            tensor_matrix_shape = tensor_fract_shape[:-3] + [
                tensor_fract_shape[-3],
                tensor_fract_shape[-2] // block_out,
                block_out,
                block_reduce
            ]
            lambda_expression = lambda *indices: tensor_fract(*indices[:-4], indices[-4],
                                                              indices[-3]*block_out + indices[-2],
                                                              indices[-1])
        tensor_matrix = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": "nd2Zn_vnchwconv"}
        )
        return tensor_matrix

    @staticmethod
    def compute_nd2nz(ori_tensor, compute_params):
        """
        reshape ori_shape's shape nd2Nz by normal way
        input_params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format is FRACTAL_Z
        """
        block_in = compute_params.get("block_in")
        block_reduce = compute_params.get("block_reduce")
        tensor_name = compute_params.get("tensor_name")
        trans = compute_params.get("trans")
        mode_info = compute_params.get("mode_info")
        if not trans:
            fract_shape, lambda_expression = FormatCompute._lambda_nd2nz(ori_tensor, block_in, block_reduce)
        else:
            fract_shape, lambda_expression = FormatCompute._lambda_nd2zn(ori_tensor, block_in, block_reduce)

        res = tvm.compute(
            fract_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info}
        )

        return res

    @staticmethod
    def compute_nz2zz_int82fp32(ori_tensor, compute_params):
        """
        reshape Nz2Zz and int8 cast to fp32
        input_params:
            ori_tensor: the tensor format is Zz ,dtype is int8
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format is FRACTAL_Z
        """
        tensor_name = compute_params.get("tensor_name")
        trans = compute_params.get("trans")
        mode_info = compute_params.get("mode_info")
        ori_shape = shape_util.shape_to_list(ori_tensor.shape)
        if not trans:
            tensor_a_normalize_shape = ori_shape[:-4] + [
                ori_shape[-3],
                ori_shape[-4] * 2,
                ori_shape[-2],
                ori_shape[-1] // 2
            ]
            lambda_expression = lambda *indices: ori_tensor(*indices[:-4], indices[-3] // 2,
                                                            indices[-4], indices[-2],
                                                            (indices[-3]*16 + indices[-1]) % 32)
        else:
            tensor_a_normalize_shape = ori_shape[:-4] + [
                ori_shape[-4] * 2,
                ori_shape[-3],
                ori_shape[-2],
                ori_shape[-1] // 2
            ]
            lambda_expression = lambda *indices: ori_tensor(*indices[:-4], indices[-4] // 2,
                                                            indices[-3], indices[-2],
                                                            (indices[-4]*16 + indices[-1]) % 32)
        res = tvm.compute(
            tensor_a_normalize_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info}
        )

        return res

    @staticmethod
    def compute_zn2zn_int82fp32(ori_tensor, compute_params):
        """
        reshape Zn-int8 to Zn-fp16
        input_params:
            ori_tensor: the tensor format is Zn ,dtype is int8
            compute_params: dict, the info need trans to schedule
        return:
            the tensor is Zn-fp16
        """
        tensor_name = compute_params.get("tensor_name")
        trans = compute_params.get("trans")
        mode_info = compute_params.get("mode_info")
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        if not trans:
            tensor_normalize_shape = ori_tensor_shape[:-4] + [
                ori_tensor_shape[-4] * 2,
                ori_tensor_shape[-3],
                ori_tensor_shape[-2],
                ori_tensor_shape[-1] // 2
            ]
            lambda_expression = lambda *indices: ori_tensor(*indices[:-4], indices[-4] // 2,
                                                            indices[-3], indices[-2],
                                                            (indices[-4]*16 + indices[-1]) % 32)
        else:
            tensor_normalize_shape = ori_tensor_shape[:-4] + [
                ori_tensor_shape[-3],
                ori_tensor_shape[-4] * 2,
                ori_tensor_shape[-1] // 2,
                ori_tensor_shape[-2]
            ]
            lambda_expression = lambda *indices: ori_tensor(*indices[:-4], indices[-3] // 2,
                                                            indices[-4], indices[-1],
                                                            (indices[-3]*16 + indices[-2]) % 32)
        res = tvm.compute(
            tensor_normalize_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info}
        )

        return res

    @staticmethod
    def compute_nz2nd(ori_tensor, output_shape=None, tensor_name="tensor_nz2nd", res_tag="", attrs_dict=None):
        """
        reshape the ori_tensor Nz to nd
        input params:
            ori_tensor: the tensor format is Nz
            output_shape: the out tensor's shape
            tensor_name: the tensor's name after reshape in IR
            res_tag: the tag used in the out tensor
            attrs_dict: dict, the info need trans to schedule
        return:
            the format is ND tensor
        """
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        block_out = ori_tensor_shape[-1]
        block_in = ori_tensor_shape[-2]
        shapes = ori_tensor_shape[:-4] + [
            ori_tensor_shape[-3]*block_in, ori_tensor_shape[-4]*block_out
        ]
        if in_dynamic():
            shape_trans = ori_tensor_shape[:-4] + [
                ori_tensor_shape[-3], block_in, ori_tensor_shape[-4], block_out
            ]
            before_c_gm = tvm.compute(shape_trans,
                                      lambda *indices: ori_tensor(*indices[:-4],
                                                                  indices[-2], indices[-4],
                                                                  indices[-3], indices[-1]),
                                      name="before_c_gm")
            lambda_expression = lambda *indices: before_c_gm(*indices[:-2],
                                                            indices[-2] // block_in,
                                                            indices[-2] % block_in,
                                                            indices[-1] // block_out,
                                                            indices[-1] % block_out)
        else:
            lambda_expression = lambda *indices: ori_tensor(*indices[:-2],
                                                             indices[-1] // block_out,
                                                             indices[-2] // block_in,
                                                             indices[-2] % block_in,
                                                             indices[-1] % block_out)
        if output_shape is not None:
            shapes = output_shape
        if attrs_dict is None:
            attrs_dict = dict()
        attrs_dict["not_reuse_pre_tensors"] = True
        if res_tag != "":
            tensor_c = tvm.compute(shapes, lambda_expression, name=tensor_name, tag=res_tag, attrs=attrs_dict)
        else:
            tensor_c = tvm.compute(shapes, lambda_expression, name=tensor_name, attrs=attrs_dict)
        return tensor_c

    @staticmethod
    def tvm_compute_nd_add_nz_to_nd(tensor_beta_bias, tensor_alpha_c, tensor_name):
        """
        compute C_matrix = A_matrix + B_matrix, A_matrix's format is ND, B_matrix's format is
        Nz, the result C_matrix's format is ND. support batch
        input args:
            tensor_beta_bias: the tensor with ND format
            tensor_alpha_c: the tensor with Nz format
            tensor_name: the result tensor's name in IR
        return:
            res: the add result, format is ND
        """
        res_shape = tensor_beta_bias.shape
        if len(res_shape) == 2:
            lambda_expression = lambda i, j: (tensor_beta_bias[i, j]
                + tensor_alpha_c[j // 16, i // 16, i % 16, j % 16])
        else:
            lambda_expression = lambda batch, i, j: (tensor_beta_bias[batch, i, j]
                + tensor_alpha_c[batch, j // 16, i // 16, i % 16, j % 16])

        res = tvm.compute(res_shape, lambda_expression, name=tensor_name)
        return res

    @staticmethod
    def tvm_compute_nd_add_nz_to_nz(tensor_beta_bias, tensor_alpha_c, tensor_name):
        """
        compute C_matrix = A_matrix + B_matrix, A_matrix's format is ND, B_matrix's format is
        Nz, the result C_matrix's format is Nz. support batch
        input args:
            tensor_beta_bias: the tensor with ND format
            tensor_alpha_c: the tensor with Nz format
            tensor_name: the result tensor's name in IR
        return:
            res: the add result, format is Nz
        """
        res_shape = tensor_alpha_c.shape
        block_in = res_shape[-2]
        block_out = res_shape[-1]
        if len(res_shape) == 4:
            lambda_expression = lambda i, j, k, l: tensor_beta_bias[j * block_in + k, i * block_out + l]\
                + tensor_alpha_c[i, j, k, l]
        else:
            lambda_expression = lambda batch, i, j, k, l: tensor_beta_bias[batch, j * block_in + k, i * block_out + l]\
                + tensor_alpha_c[batch, i, j, k, l]

        res = tvm.compute(res_shape, lambda_expression, name=tensor_name)
        return res

    @staticmethod
    def compute_nd2zz_vnchwconv(ori_tensor, compute_params):
        """
        use vnchwconv to reshape nd shape: (m, k) --> (m/m0, k, m0) --> (m/m0, k/k0, m0, k0)
        inpute params:
            ori_tensor: tensor, format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format is FRACTAL_Z
        """
        tensor_name = compute_params.get("tensor_name")
        block_in = compute_params.get("block_in")
        block_reduce = compute_params.get("block_reduce")
        trans = compute_params.get("trans")
        mode_info = compute_params.get("mode_info")
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        tensor_fract_k_shape = ori_tensor_shape[:-2] + [
            int_ceil_div(ori_tensor_shape[-2], block_in),
            ori_tensor_shape[-1],
            block_in
        ]
        tensor_fract_k = tvm.compute(
            tensor_fract_k_shape,
            lambda *indices: ori_tensor(*indices[:-3], indices[-3]*block_in + indices[-1], indices[-2]),
            name="{}_fract_k".format(tensor_name)
        )
        if trans:
            tensor_matrix_shape = tensor_fract_k_shape[:-3] + [
                int_ceil_div(tensor_fract_k_shape[-2], block_reduce),
                tensor_fract_k_shape[-3],
                block_reduce,
                block_in
            ]
            lambda_expression = lambda *indices: tensor_fract_k(*indices[:-4], indices[-3],
                                                                indices[-4]*block_in + indices[-2],
                                                                indices[-1])
        else:
            tensor_matrix_shape = tensor_fract_k_shape[:-3] + [
                tensor_fract_k_shape[-3],
                int_ceil_div(tensor_fract_k_shape[-2], block_reduce),
                block_in,
                block_reduce
            ]
            lambda_expression = lambda *indices: tensor_fract_k(*indices[:-4], indices[-4],
                                                                indices[-3]*block_reduce + indices[-1],
                                                                indices[-2])
        res = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info}
        )
        return res

    @staticmethod
    def _trans_pose(ori_shape, perm):
        """
        get the shape and lambda upon on the perm
        """
        dst_indice = ['batch', 'i', 'j', 'k', 'l'][-len(ori_shape):]
        if len(perm) > len(ori_shape):
            perm = list(perm_num - 1 for perm_num in perm[-len(ori_shape):])
        src_indice = list(dst_indice[indice] for indice in perm)
        shape = list(ori_shape[indice] for indice in perm)
        dst_indice = ",".join(dst_indice)
        src_indice = ",".join(src_indice)
        lambda_expression_str = f'lambda {dst_indice}: ori_tensor[{src_indice}]'
        return shape, lambda_expression_str

    @staticmethod
    def _get_var_name(tensor_format, var_name, cache_tiling_flag):
        ori_flag = (tensor_format == "ND" and not cache_tiling_flag) or (var_name == "k" and cache_tiling_flag)
        if ori_flag:
            var_name += "_ori"
        return var_name

    @staticmethod
    def _lambda_nd2nz(ori_tensor, block_in, block_reduce):
        """
        the express of nd format trans to Nz format. support batch
        input args:
            ori_tensor: the tensor need trans to Nz
            block_in: the block_in of chip
            block_reduce: the block_reduce of chip
        return:
            fract_shape: the shape after reshape
            lambda_expression the expression of trans nd to nz
        """
        ori_nd_shape = shape_util.shape_to_list(ori_tensor.shape)
        fract_shape = ori_nd_shape[:-2] + [
            int_ceil_div(ori_nd_shape[-1], block_reduce),
            int_ceil_div(ori_nd_shape[-2], block_in),
            block_in,
            block_reduce
        ]
        if len(ori_nd_shape) == 3:
            lambda_expression = lambda batch, k1, n1, n0, k0: ori_tensor[batch, n1 * block_in + n0,
                                k1 * block_reduce + k0]
        else:
            lambda_expression = lambda k1, n1, n0, k0: ori_tensor[n1 * block_in + n0, k1 * block_reduce + k0]
        return fract_shape, lambda_expression

    @staticmethod
    def _lambda_nd2zn(ori_tensor, block_in, block_reduce):
        """
        the express of nd format trans to Zz format. support batch
        input args:
            ori_tensor: the tensor need trans to Nz
            block_in: the block_in of chip
            block_reduce: the block_reduce of chip
        return:
            fract_shape: the shape after reshape
            lambda_expression the expression of trans nd to nz
        """
        ori_nd_shape = shape_util.shape_to_list(ori_tensor.shape)
        fract_shape = ori_nd_shape[:-2] + [
            int_ceil_div(ori_nd_shape[-2], block_reduce),
            int_ceil_div(ori_nd_shape[-1], block_in),
            block_in,
            block_reduce
        ]
        fract_l1_shape = fract_shape[:-2] + [block_reduce, block_in]
        if len(ori_nd_shape) == 3:
            tensor_l1 = tvm.compute(fract_l1_shape, lambda batch, k1, n1, k0, n0:
                                    ori_tensor[batch, k1 * block_reduce + k0, n1 * block_in + n0],
                                    name="tensor_b_l1")
            lambda_expression = lambda batch, k1, n1, n0, k0: tensor_l1[batch, k1, n1, k0, n0]
        else:
            tensor_l1 = tvm.compute(fract_l1_shape, lambda k1, n1, k0, n0:
                                    ori_tensor[k1 * block_reduce + k0, n1 * block_in + n0],
                                    name="tensor_b_l1")
            lambda_expression = lambda k1, n1, n0, k0: tensor_l1[k1, n1, k0, n0]

        return fract_shape, lambda_expression

    @staticmethod
    def _get_shape_by_batch(shapes, tensor_a_shape, tensor_b_shape, batch_max, batch_flags):
        both_with_batch = batch_flags[0]
        a_with_batch = batch_flags[1]
        b_with_batch = batch_flags[2]
        if batch_max:
            batch_shape = functools_reduce(lambda x, y: x * y, batch_max)
            shapes.insert(0, batch_shape)
        elif both_with_batch or a_with_batch:
            batch_shape = tensor_a_shape[0]
            shapes.insert(0, batch_shape)
        elif b_with_batch:
            batch_shape = tensor_b_shape[0]
            shapes.insert(0, batch_shape)

    @staticmethod
    def _choose_mmad_compute_func(batch_flags):
        both_with_batch = batch_flags[0]
        a_with_batch = batch_flags[1]
        b_with_batch = batch_flags[2]
        if both_with_batch:
            mad_compute_func = FormatCompute._mad_both_batch
        elif a_with_batch:
            mad_compute_func = FormatCompute._mad_a_batch
        elif b_with_batch:
            mad_compute_func = FormatCompute._mad_b_batch
        else:
            mad_compute_func = FormatCompute._mad_none_batch
        return mad_compute_func

    @staticmethod
    def _batch_broadcast(tensor_a, tensor_b):
        """
        broadcast batch_a and batch_b

        Parameters:
            tensor_a: tvm tensor, fp16
            tensor_b: tvm tensor, fp16

        Returns:
            batch_max: the result of broadcast, list
            batch_a: unsqueezed batch of tensor_a, list
            batch_b: unsqueezed batch of tensor_b, list
        """
        batch_max = None
        batch_a = []
        batch_b = []
        if "ori_batch_shape" in tensor_a.op.attrs and "ori_batch_shape" in tensor_b.op.attrs:
            ori_batch_a = tensor_a.op.attrs["ori_batch_shape"]
            ori_batch_b = tensor_b.op.attrs["ori_batch_shape"]
            batch_a = shape_util.shape_to_list(ori_batch_a)
            batch_b = shape_util.shape_to_list(ori_batch_b)
        batch_a_dim = len(batch_a)
        batch_b_dim = len(batch_b)
        is_batch_broadcast = (batch_a_dim > 0 and batch_b_dim > 0 and batch_a != batch_b)

        if is_batch_broadcast:
            batch_a, batch_b, batch_max = broadcast_shapes(batch_a, batch_b)

        return batch_max, batch_a, batch_b

    @staticmethod
    def _mapping_batch_a_or_b(batch_reduce, batch_max, batch_ori):
        """
        map batch_reduce into batch_a or batch_b
        e.g. A x B = C
            A = [B1,1,1,B4, K//16,M//16,16,16]
            B = [B2,B3,1, N//16,K//16,16,16]
            C = [B1,B2,B3,B4, N//16,M//16,16,16]

            batch_reduce is in range of 0 to B1*B2*B3*B4-1,
            b1 = batch_reduce // (B2*B3*B4),
            b2 = batch_reduce // (B3*B4) % B2,
            b3 = batch_reduce // B4 % B3,
            b4 = batch_reduce % B4

            batch_a = B4*b1 + b4,
            batch_b = B3*b2 + b3

        Parameters:
            batch_reduce: reduce shape of batch_max, number
            batch_max: broadcast batch, list
            batch_ori: original batch_a or batch_b, list

        Returns:
            batch_mapping: the right batch in batch_a or batch_b
                corresponding to batch_reduce
        """
        batch_mapping = 0
        for i, val in enumerate(batch_ori):
            if val != 1:
                if i == len(batch_ori) - 1:
                    batch_tmp = batch_reduce % batch_max[i]
                elif i == 0:
                    batch_tmp = batch_reduce // functools_reduce(lambda x, y: x * y, batch_max[i+1:])
                else:
                    batch_tmp = batch_reduce // functools_reduce(lambda x, y: x * y, batch_max[i+1:]) % batch_max[i]
                if i == len(batch_ori) - 1:
                    batch_mapping += batch_tmp
                else:
                    batch_mapping += batch_tmp * functools_reduce(lambda x, y: x * y, batch_ori[i+1:])
        return batch_mapping

    @staticmethod
    def _mad_both_batch(tensor_a, tensor_b, reduce_axis,
        matrix_type, offset_x, block_out, need_add_bias_in_fb, bias_fb=None):
        reduce_kb = reduce_axis[0]
        reduce_kp = reduce_axis[1]
        batch_max, batch_a, batch_b = FormatCompute._batch_broadcast(tensor_a, tensor_b)
        if not need_add_bias_in_fb:
            if batch_max:
                lambda_expression = lambda batch, nb, mb, mp, np: tvm.sum(
                    ((tensor_a[FormatCompute._mapping_batch_a_or_b(batch, batch_max, batch_a),
                        mb, reduce_kb, mp, reduce_kp] - offset_x) *
                    tensor_b[FormatCompute._mapping_batch_a_or_b(batch, batch_max, batch_b),
                        reduce_kb, nb, np, reduce_kp]).astype(matrix_type),
                    axis=[reduce_kb, reduce_kp])
            else:
                lambda_expression = lambda batch, nb, mb, mp, np: tvm.sum(
                    ((tensor_a[batch, mb, reduce_kb, mp, reduce_kp] - offset_x)
                        * tensor_b[batch, reduce_kb, nb, np, reduce_kp]).astype(matrix_type),
                    axis=[reduce_kb, reduce_kp])
        else:
            lambda_expression = lambda batch, nb, mb, mp, np: tvm.sum(
                tvm.select(tvm.all(reduce_kb.var == 0, reduce_kp.var == 0),
                    ((tensor_a[batch, mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        batch, reduce_kb, nb, np, reduce_kp]).astype(
                            matrix_type) +
                    bias_fb[nb * block_out + np],
                    ((tensor_a[batch, mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        batch, reduce_kb, nb, np, reduce_kp]).astype(
                            matrix_type)),
                axis=[reduce_kb, reduce_kp])
        return lambda_expression

    @staticmethod
    def _mad_a_batch(tensor_a, tensor_b, reduce_axis,
        matrix_type, offset_x, block_out, need_add_bias_in_fb, bias_fb=None):
        reduce_kb = reduce_axis[0]
        reduce_kp = reduce_axis[1]
        if not need_add_bias_in_fb:
            lambda_expression = lambda batch, nb, mb, mp, np: tvm.sum(
                ((tensor_a[batch, mb, reduce_kb, mp, reduce_kp] - offset_x)
                    * tensor_b[reduce_kb, nb, np, reduce_kp]).astype(matrix_type),
                axis=[reduce_kb, reduce_kp])
        else:
            lambda_expression = lambda batch, nb, mb, mp, np: tvm.sum(
                tvm.select(tvm.all(reduce_kb.var == 0, reduce_kp.var == 0),
                    ((tensor_a[batch, mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        reduce_kb, nb, np, reduce_kp]).astype(
                            matrix_type) +
                    bias_fb[nb * block_out + np],
                    ((tensor_a[batch, mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        reduce_kb, nb, np, reduce_kp]).astype(
                            matrix_type)),
                axis=[reduce_kb, reduce_kp])
        return lambda_expression

    @staticmethod
    def _mad_b_batch(tensor_a, tensor_b, reduce_axis,
        matrix_type, offset_x, block_out, need_add_bias_in_fb, bias_fb=None):
        reduce_kb = reduce_axis[0]
        reduce_kp = reduce_axis[1]
        if not need_add_bias_in_fb:
            lambda_expression = lambda batch, nb, mb, mp, np: tvm.sum(
                ((tensor_a[mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                    batch, reduce_kb, nb, np, reduce_kp]).astype(matrix_type),
                axis=[reduce_kb, reduce_kp])
        else:
            lambda_expression = lambda batch, nb, mb, mp, np: tvm.sum(
                tvm.select(tvm.all(reduce_kb.var == 0, reduce_kp.var == 0),
                    ((tensor_a[mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        batch, reduce_kb, nb, np, reduce_kp]).astype(
                            matrix_type) + bias_fb[nb * block_out + np],
                    ((tensor_a[mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        batch, reduce_kb, nb, np, reduce_kp]).astype(
                            matrix_type)),
                axis=[reduce_kb, reduce_kp])
        return lambda_expression

    @staticmethod
    def _mad_none_batch(tensor_a, tensor_b, reduce_axis,
        matrix_type, offset_x, block_out, need_add_bias_in_fb, bias_fb=None):
        reduce_kb = reduce_axis[0]
        reduce_kp = reduce_axis[1]
        if not need_add_bias_in_fb:
            lambda_expression = lambda nb, mb, mp, np: tvm.sum(
                ((tensor_a[mb, reduce_kb, mp, reduce_kp] - offset_x) * (tensor_b[
                    reduce_kb, nb, np, reduce_kp])).astype(matrix_type),
                axis=[reduce_kb, reduce_kp])
        else:
            lambda_expression = lambda nb, mb, mp, np: tvm.sum(
                tvm.select(tvm.all(reduce_kb.var == 0, reduce_kp.var == 0),
                    ((tensor_a[mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        reduce_kb, nb, np, reduce_kp]).astype(matrix_type) +
                        bias_fb[nb * block_out + np],
                    ((tensor_a[mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        reduce_kb, nb, np, reduce_kp]).astype(matrix_type)),
                axis=[reduce_kb, reduce_kp])
        return lambda_expression

    @staticmethod
    def _compute_nd2zz_int8_trans(ori_tensor, compute_params):
        """
        this func is reshape nd2Zz when dtype is int8 and trans and ori_shape is ([m, k]
        input params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format is FRACTAL_Z
        """
        block_in = compute_params.get("block_in")
        block_reduce = compute_params.get("block_reduce")
        mode_info = compute_params.get("mode_info")
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        ori_tensor_shape[-1], ori_tensor_shape[-2] = ori_tensor_shape[-2], ori_tensor_shape[-1]
        tensor_matrix_shape = ori_tensor_shape[:-2] + [
            ori_tensor_shape[-2] // block_in,
            ori_tensor_shape[-1] // block_reduce,
            block_in,
            block_reduce
        ]
        tensor_a_transpose = tvm.compute(
            ori_tensor_shape,
            lambda *indices: ori_tensor(*indices[:-2], indices[-1], indices[-2]),
            name="tensor_a_transpose")
        lambda_expression = lambda *indices: tensor_a_transpose(*indices[:-4],
                                                                indices[-4]*block_in + indices[-2],
                                                                indices[-3]*block_reduce + indices[-1])
        res = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name="tensor_a_zz",
            attrs={"mode": mode_info})

        return res

    @staticmethod
    def _compute_nd2zn_int8_no_trans(ori_tensor, compute_params):
        """
        this func is reshape nd2Zn when dtype is int8 and not trans and ori_shape is [k, n]
        input params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format is FRACTAL_Z
        """
        block_out = compute_params.get("block_out")
        block_reduce = compute_params.get("block_reduce")
        mode_info = compute_params.get("mode_info")

        normalize_shape = shape_util.shape_to_list(ori_tensor.shape)
        normalize_shape[-1], normalize_shape[-2] = normalize_shape[-2], normalize_shape[-1]
        tensor_fract_shape = normalize_shape[:-2] + [
            normalize_shape[-1] // block_reduce,
            normalize_shape[-2] // block_out,
            block_out,
            block_reduce
        ]
        tensor_transpose = tvm.compute(
            normalize_shape,
            lambda *indices: ori_tensor(*indices[:-2], indices[-1], indices[-2]),
            name="tensor_b_transpose"
        )
        res = tvm.compute(
            tensor_fract_shape,
            lambda *indices: tensor_transpose(*indices[:-4],
                                              indices[-3]*block_out + indices[-2],
                                              indices[-4]*block_reduce + indices[-1]),
            name="tensor_b_zn",
            attrs={"mode": mode_info}
        )
        return res

    def tvm_compute_mad(self, tensor_a, tensor_b, tensor_bias, reduce_kb, reduce_kp,
        matrix_type, mad_mode, tensor_name, need_add_bias_in_fb, offset_x, attrs_dict):
        """
        compute A x B add bias, support batch
        input params:
            tensor_a: the tensor A
            tensor_b: the tensor B
            tensor_bias: the bias tensor
            reduce_kb: reduce axis k1
            reduce_kp: reduce axis k0
            matrix_type: the dtype of the mmad tensor
            mad_mode: str, gemv/gevm/gemm
            tensor_name: str, the mmad tensor's name
            need_add_bias_in_fb: if add bias in fixpipe buffer
            offset_x: the offset tensor
            attrs_dict: dict, the info need trans to schedule
        return:
            the mmad tensor
        """
        tensor_a_shape = shape_util.shape_to_list(tensor_a.shape)
        tensor_b_shape = shape_util.shape_to_list(tensor_b.shape)
        a_with_batch = len(tensor_a_shape) == 5
        b_with_batch = len(tensor_b_shape) == 5
        both_with_batch = a_with_batch and b_with_batch

        block_in = tensor_a_shape[-2] if mad_mode != "gemv" else tensor_b_shape[-2]
        block_out = tensor_b_shape[-2] if mad_mode != "gemv" else tensor_a_shape[-2]

        # this is equivalent to alignment in the compute phase
        block_in, block_out = [16, 16] if mad_mode in ("gemv", "gevm") else [block_in, block_out]
        tensor_a, tensor_b = [tensor_b, tensor_a] if mad_mode == "gemv" else [tensor_a, tensor_b]
        m_shape = tensor_a_shape[-4] if mad_mode != "gemv" else tensor_b_shape[-4]
        n_shape = tensor_b_shape[-3] if mad_mode != "gemv" else tensor_a_shape[-3]
        shapes = [n_shape, m_shape, block_in, block_out]
        batch_flags = [both_with_batch, a_with_batch, b_with_batch]
        batch_max, _, _ = self._batch_broadcast(tensor_a, tensor_b)
        self._get_shape_by_batch(shapes, tensor_a_shape, tensor_b_shape, batch_max, batch_flags)
        reduce_axis = [reduce_kb, reduce_kp]

        mad_compute_func = self._choose_mmad_compute_func(batch_flags)
        lambda_expression = mad_compute_func(tensor_a, tensor_b, reduce_axis,
            matrix_type, offset_x, block_out, need_add_bias_in_fb, tensor_bias)

        tensor_c = tvm.compute(shapes, lambda_expression, name=tensor_name, attrs=attrs_dict)

        return tensor_c

    def fract_change_outer_axis(self, ori_tensor, compute_params):
        """
        change_axis [batch, i, j, k, l] -> [batch, j, i, k, l]
        input_args:
            ori_tensor: fractal format tensor
            compute_params: dict, the info need trans to schedule
        return:
            compute result
        """
        tensor_name = compute_params.get("tensor_name")
        trans = compute_params.get("trans")
        mode_info = compute_params.get("mode_info")
        if trans:
            compute_params["trans"] = False
            res = self.fract_change_inner_axis(ori_tensor, compute_params)
            return res

        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        perm = (0, 2, 1, 3, 4)
        shapes, lambda_expression_str = self._trans_pose(ori_tensor_shape, perm)

        res = tvm.compute(shapes, eval(lambda_expression_str, locals()), name=tensor_name,
            attrs={"mode": mode_info})
        return res

    def fract_change_inner_axis(self, ori_tensor, compute_params):
        """
        change_axis [batch, i, j, k, l] -> [batch, i, j, l, k]
        input_args:
            ori_tensor: fractal format tensor
            compute_params: dict, the info need trans to schedule
        return:
            compute result
        """
        tensor_name = compute_params.get("tensor_name")
        trans = compute_params.get("trans")
        mode_info = compute_params.get("mode_info")
        if trans:
            compute_params["trans"] = False
            res = self.fract_change_outer_axis(ori_tensor, compute_params)
            return res
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        perm = (0, 1, 2, 4, 3)
        shapes, lambda_expression_str = self._trans_pose(ori_tensor_shape, perm)

        res = tvm.compute(shapes, eval(lambda_expression_str, locals()), name=tensor_name,
            attrs={"mode": mode_info})
        return res

    def fract_change_both_axis(self, ori_tensor, compute_params):
        """
        change_axis [batch, i, j, k, l] -> [batch, j, i, l, k]
        input_args:
            ori_tensor: fractal format tensor
            compute_params: dict, the info need trans to schedule
        return:
            compute result
        """
        tensor_name = compute_params.get("tensor_name")
        trans = compute_params.get("trans")
        mode_info = compute_params.get("mode_info")
        ori_tensor_shape = shape_util.shape_to_list(ori_tensor.shape)
        if trans:
            shapes = ori_tensor_shape
            lambda_expression = lambda *indices: ori_tensor[indices]
        else:
            perm = (0, 2, 1, 4, 3)
            shapes, lambda_expression_str = self._trans_pose(ori_tensor_shape, perm)
            lambda_expression = eval(lambda_expression_str, locals())

        res = tvm.compute(shapes, lambda_expression, name=tensor_name, attrs={"mode": mode_info})
        return res
