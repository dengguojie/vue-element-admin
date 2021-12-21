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
    def _get_value(shape_object):
        """
        get the value if shape_object when having attr "value"
        """
        return shape_object.value if hasattr(shape_object, "value") else shape_object

    def _lambda_nd2Nz(self, ori_tensor, block_in, block_reduce):
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
        ori_nd_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_nd_shape) == 3:
            fract_shape = (
                ori_nd_shape[0],
                int_ceil_div(ori_nd_shape[2], block_reduce),
                int_ceil_div(ori_nd_shape[1], block_in),
                block_in,
                block_reduce
            )
            lambda_expression = lambda batch, k1, n1, n0, k0: ori_tensor[batch, n1 * block_in + n0, k1 * block_reduce + k0]
        else:
            fract_shape = (
                int_ceil_div(ori_nd_shape[1], block_reduce),
                int_ceil_div(ori_nd_shape[0], block_in),
                block_in,
                block_reduce
            )
            lambda_expression = lambda k1, n1, n0, k0: ori_tensor[n1 * block_in + n0, k1 * block_reduce + k0]
        return fract_shape, lambda_expression

    def _lambda_nd2Zn(self, ori_tensor, block_in, block_reduce):
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
        ori_nd_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_nd_shape) == 3:
            fract_shape = (
                ori_nd_shape[0],
                int_ceil_div(ori_nd_shape[1], block_reduce),
                int_ceil_div(ori_nd_shape[2], block_in),
                block_in,
                block_reduce,
            )
            fract_L1_shape = (
                ori_nd_shape[0],
                int_ceil_div(ori_nd_shape[1], block_reduce),
                int_ceil_div(ori_nd_shape[2], block_in),
                block_reduce,
                block_in,
            )
            tensor_l1 = tvm.compute(fract_L1_shape, lambda batch, k1, n1, k0, n0:
                                    ori_tensor[batch, k1 * block_reduce + k0, n1 * block_in + n0],
                                    name="tensor_b_l1")
            lambda_expression = lambda batch, k1, n1, n0, k0: tensor_l1[batch, k1, n1, k0, n0]
        else:
            fract_shape = (
                int_ceil_div(ori_nd_shape[0], block_reduce),
                int_ceil_div(ori_nd_shape[1], block_in),
                block_in,
                block_reduce,
            )
            fract_L1_shape = (
                int_ceil_div(ori_nd_shape[0], block_reduce),
                int_ceil_div(ori_nd_shape[1], block_in),
                block_reduce,
                block_in,
            )
            tensor_l1 = tvm.compute(fract_L1_shape, lambda k1, n1, k0, n0:
                                    ori_tensor[k1 * block_reduce + k0, n1 * block_in + n0],
                                    name="tensor_b_l1")
            lambda_expression = lambda k1, n1, n0, k0: tensor_l1[k1, n1, k0, n0]

        return fract_shape, lambda_expression

    def tvm_compute_nd_add_Nz_to_nd(self, tensor_beta_bias, tensor_alpha_c, tensor_name):
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

    def tvm_compute_nd_add_Nz_to_Nz(self, tensor_beta_bias, tensor_alpha_c, tensor_name):
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

    def tvm_compute_mad(self, tensor_a, tensor_b, tensor_bias, reduce_kb, reduce_kp,
        matrix_type, mad_mode, tensor_name, need_add_bias_in_fb, offset_x, attrs_dict):
        """
        compute A x B add bias, support batch
        """
        tensor_a_shape = [self._get_value(i) for i in tensor_a.shape]
        tensor_b_shape = [self._get_value(i) for i in tensor_b.shape]
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

    def _get_shape_by_batch(self, shapes, tensor_a_shape, tensor_b_shape, batch_max, batch_flags):
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

    def _choose_mmad_compute_func(self, batch_flags):
        both_with_batch = batch_flags[0]
        a_with_batch = batch_flags[1]
        b_with_batch = batch_flags[2]
        if both_with_batch:
            mad_compute_func = self._mad_both_batch
        elif a_with_batch:
            mad_compute_func = self._mad_a_batch
        elif b_with_batch:
            mad_compute_func = self._mad_b_batch
        else:
            mad_compute_func = self._mad_none_batch
        return mad_compute_func

    def _batch_broadcast(self, tensor_a, tensor_b):
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
            if ori_batch_a != "none":
                batch_a = [self._get_value(i) for i in ori_batch_a]
            if ori_batch_b != "none":
                batch_b = [self._get_value(i) for i in ori_batch_b]
        batch_a_dim = len(batch_a)
        batch_b_dim = len(batch_b)
        is_batch_broadcast = (batch_a_dim > 0 and batch_b_dim > 0 and batch_a != batch_b)

        if is_batch_broadcast:
            batch_a, batch_b, batch_max = broadcast_shapes(batch_a, batch_b)

        return batch_max, batch_a, batch_b

    def _mapping_batch_a_or_b(self, batch_reduce, batch_max, batch_ori):
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
        for i in range(len(batch_ori)):
            if batch_ori[i] != 1:
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

    def _mad_both_batch(self, tensor_a, tensor_b, reduce_axis,
        matrix_type, offset_x, block_out, need_add_bias_in_fb, bias_fb=None):
        reduce_kb = reduce_axis[0]
        reduce_kp = reduce_axis[1]
        batch_max, batch_a, batch_b = self._batch_broadcast(tensor_a, tensor_b)
        if not need_add_bias_in_fb:
            if batch_max:
                lambda_expression = lambda batch, nb, mb, mp, np: tvm.sum(
                    ((tensor_a[self._mapping_batch_a_or_b(batch, batch_max, batch_a),
                        mb, reduce_kb, mp, reduce_kp] - offset_x) *
                    tensor_b[self._mapping_batch_a_or_b(batch, batch_max, batch_b),
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

    def _mad_a_batch(self, tensor_a, tensor_b, reduce_axis,
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

    def _mad_b_batch(self, tensor_a, tensor_b, reduce_axis,
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
                            matrix_type) +
                    bias_fb[nb * block_out + np],
                    ((tensor_a[mb, reduce_kb, mp, reduce_kp] - offset_x) * tensor_b[
                        batch, reduce_kb, nb, np, reduce_kp]).astype(
                            matrix_type)),
                axis=[reduce_kb, reduce_kp])
        return lambda_expression

    def _mad_none_batch(self, tensor_a, tensor_b, reduce_axis,
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
        format_info = compute_params.get("format_info")
        ori_batch_shape = compute_params.get("ori_batch_shape", "none")
        if trans:
            compute_params["trans"] = False
            res = self.fract_change_inner_axis(ori_tensor, compute_params)
            return res

        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_tensor_shape) == 5:
            shapes = (
                ori_tensor_shape[0],
                ori_tensor_shape[2],
                ori_tensor_shape[1],
                ori_tensor_shape[3],
                ori_tensor_shape[4]
            )
            lambda_expression = lambda batch, i, j, k, l: ori_tensor[batch, j, i, k, l]
        else:
            shapes = (
                ori_tensor_shape[1],
                ori_tensor_shape[0],
                ori_tensor_shape[2],
                ori_tensor_shape[3]
            )
            lambda_expression = lambda i, j, k, l: ori_tensor[j, i, k, l]

        res = tvm.compute(shapes, lambda_expression, name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info, "ori_batch_shape": ori_batch_shape})
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
        format_info = compute_params.get("format_info")
        ori_batch_shape = compute_params.get("ori_batch_shape", "none")
        if trans:
            compute_params["trans"] = False
            res = self.fract_change_outer_axis(ori_tensor, compute_params)
            return res
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_tensor_shape) == 5:
            shapes = (
                ori_tensor_shape[0],
                ori_tensor_shape[1],
                ori_tensor_shape[2],
                ori_tensor_shape[4],
                ori_tensor_shape[3]
            )
            lambda_expression = lambda batch, i, j, k, l: ori_tensor[batch, i, j, l, k]
        else:
            shapes = (
                ori_tensor_shape[0],
                ori_tensor_shape[1],
                ori_tensor_shape[3],
                ori_tensor_shape[2]
            )
            lambda_expression = lambda i, j, k, l: ori_tensor[i, j, l, k]

        res = tvm.compute(shapes, lambda_expression, name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info, "ori_batch_shape": ori_batch_shape})
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
        format_info = compute_params.get("format_info")
        ori_batch_shape = compute_params.get("ori_batch_shape", "none")
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if trans:
            shapes = ori_tensor_shape
            lambda_expression = lambda *indices: ori_tensor[indices]
        else:
            if len(ori_tensor_shape) == 5:
                shapes = (
                    ori_tensor_shape[0],
                    ori_tensor_shape[2],
                    ori_tensor_shape[1],
                    ori_tensor_shape[4],
                    ori_tensor_shape[3]
                )
                lambda_expression = lambda batch, i, j, k, l: ori_tensor[batch, j, i, l, k]
            else:
                shapes = (
                    ori_tensor_shape[1],
                    ori_tensor_shape[0],
                    ori_tensor_shape[3],
                    ori_tensor_shape[2]
                )
                lambda_expression = lambda i, j, k, l: ori_tensor[j, i, l, k]

        if mode_info:
            res = tvm.compute(
                shapes,
                lambda_expression,
                name=tensor_name,
                attrs={"mode": mode_info, "format_info": format_info, "ori_batch_shape": ori_batch_shape})
        else:
            res = tvm.compute(shapes, lambda_expression, name=tensor_name,
                attrs={"ori_batch_shape": ori_batch_shape})
        return res

    def compute_nd2Zz_vnchwconv(self, ori_tensor, compute_params):
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
        mode_info = compute_params.get("mode_info")
        format_info = compute_params.get("format_info")
        trans = compute_params.get("trans")
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_tensor_shape) == 2:
            tensor_fract_k_shape = (
                (ori_tensor_shape[0] + block_in - 1) // block_in,
                ori_tensor_shape[1],
                block_in
            )
            tensor_fract_k = tvm.compute(
                tensor_fract_k_shape,
                lambda i, j, k: ori_tensor[i * block_in + k, j],
                name="{}_fract_k".format(tensor_name)
            )
            if trans:
                tensor_matrix_shape = (
                    (tensor_fract_k_shape[1] + block_reduce - 1) // block_reduce,
                    tensor_fract_k_shape[0],
                    block_reduce,
                    block_in
                )
                lambda_expression = lambda i, j, k, l: tensor_fract_k[j, i * block_in + k, l]
            else:
                tensor_matrix_shape = (
                    tensor_fract_k_shape[0],
                    (tensor_fract_k_shape[1] + block_reduce - 1) // block_reduce,
                    block_in,
                    block_reduce
                )
                lambda_expression = lambda i, j, k, l: tensor_fract_k[i, j * block_reduce + l, k]
        else:
            tensor_fract_k_shape = (
                ori_tensor_shape[0],
                (ori_tensor_shape[1] + block_in - 1) // block_in,
                ori_tensor_shape[2],
                block_in
            )
            tensor_fract_k = tvm.compute(
                tensor_fract_k_shape,
                lambda batch, i, j, k: ori_tensor[batch, i * block_in + k, j],
                name="{}_fract_k".format(tensor_name)
            )
            if trans:
                tensor_matrix_shape = (
                    tensor_fract_k_shape[0],
                    (tensor_fract_k_shape[2] + block_reduce - 1) // block_reduce,
                    tensor_fract_k_shape[1],
                    block_reduce,
                    block_in
                )
                lambda_expression = lambda batch, i, j, k, l: tensor_fract_k[batch, j, i * block_in + k, l]
            else:
                tensor_matrix_shape = (
                    tensor_fract_k_shape[0],
                    tensor_fract_k_shape[1],
                    (tensor_fract_k_shape[2] + block_reduce - 1) // block_reduce,
                    block_in,
                    block_reduce
                )
                lambda_expression = lambda batch, i, j, k, l: tensor_fract_k[batch, i, j * block_reduce + l, k]

        res = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info}
        )
        return res

    def _compute_nd2Zz_int8_trans(self, ori_tensor, compute_params):
        """
        this func is reshape nd2Zz when dtype is int8 and trans and ori_shape is ([m, k]
        input params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format is FRACTAL_Z
        """
        tensor_name = compute_params.get("tensor_name")
        block_in = compute_params.get("block_in")
        block_reduce = compute_params.get("block_reduce")
        mode_info = compute_params.get("mode_info")
        format_info = compute_params.get("format_info")
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_tensor_shape) == 2:
            ori_tensor_shape[-1], ori_tensor_shape[-2] = ori_tensor_shape[-2], ori_tensor_shape[-1]
            tensor_a_transpose = tvm.compute(
                ori_tensor_shape,
                lambda i, j: ori_tensor[j, i],
                name="a_transpose")
            tensor_matrix_shape = (
                ori_tensor_shape[0] // block_in,
                ori_tensor_shape[1] // block_reduce,
                block_in,
                block_reduce
            )
            lambda_expression = lambda i, j, k, l: tensor_a_transpose[i * block_in + k, j * block_reduce + l]
        else:
            ori_tensor_shape[-1], ori_tensor_shape[-2] = ori_tensor_shape[-2], ori_tensor_shape[-1]
            tensor_a_transpose = tvm.compute(
                ori_tensor_shape,
                lambda batch, i, j: ori_tensor[batch, j, i],
                name="a_transpose")
            tensor_matrix_shape = (
                ori_tensor_shape[0],
                ori_tensor_shape[1] // block_in,
                ori_tensor_shape[2] // block_reduce,
                block_in,
                block_reduce
            )
            lambda_expression = lambda batch, i, j, k, l: tensor_a_transpose[batch,
                                                                             i * block_in + k, j * block_reduce + l]

        res = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name="tensor_a_matrix",
            attrs={"mode": mode_info, "format_info": format_info})

        return res

    def compute_nd2Zz(self, ori_tensor, compute_params):
        """
        reshape nd2Zz by normal way
        input params:
            ori_tensor: the tensor format is nd
        return:
           the tensor format Zz
        """
        tensor_name = compute_params.get("tensor_name")
        block_in = compute_params.get("block_in")
        block_reduce = compute_params.get("block_reduce")
        data_flow = compute_params.get("data_flow")
        mode_info = compute_params.get("mode_info", "none")
        format_info = compute_params.get("format_info", "none")
        trans = compute_params.get("trans")
        int82int32_trans_flag = (data_flow == "int82int32") and trans
        if int82int32_trans_flag:
            return self._compute_nd2Zz_int8_trans(ori_tensor, compute_params)
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_tensor_shape) == 2:
            if trans:
                tensor_matrix_shape = (
                    int_ceil_div(ori_tensor_shape[1], block_in),
                    int_ceil_div(ori_tensor_shape[0], block_reduce),
                    block_in,
                    block_reduce
                )

                tensor_L1_shape = (
                    int_ceil_div(ori_tensor_shape[1], block_in),
                    int_ceil_div(ori_tensor_shape[0], block_reduce),
                    block_reduce,
                    block_in
                )
                L1_tensor = tvm.compute(tensor_L1_shape,
                            lambda m1, k1, k0, m0: ori_tensor[k1 * block_reduce + k0, m1 * block_in + m0],
                            name="tensor_a_l1")
                lambda_expression = lambda m1, k1, m0, k0: L1_tensor[m1, k1, k0, m0]
            else:
                tensor_matrix_shape = (
                    int_ceil_div(ori_tensor_shape[0], block_in),
                    int_ceil_div(ori_tensor_shape[1], block_reduce),
                    block_in,
                    block_reduce
                )
                lambda_expression = lambda m1, k1, m0, k0: ori_tensor[m1 * block_in + m0, k1 * block_reduce + k0]
        else:
            if trans:
                tensor_matrix_shape = (
                    ori_tensor_shape[0],
                    int_ceil_div(ori_tensor_shape[2], block_in),
                    int_ceil_div(ori_tensor_shape[1], block_reduce),
                    block_in,
                    block_reduce
                )
                tensor_L1_shape = (
                    ori_tensor_shape[0],
                    int_ceil_div(ori_tensor_shape[2], block_in),
                    int_ceil_div(ori_tensor_shape[1], block_reduce),
                    block_reduce,
                    block_in
                )
                L1_tensor = tvm.compute(tensor_L1_shape,
                            lambda batch, m1, k1, k0, m0: ori_tensor[batch, k1 * block_reduce + k0,
                                m1 * block_in + m0], name="tensor_a_l1")
                lambda_expression = lambda batch, m1, k1, m0, k0:L1_tensor[batch, m1, k1, k0, m0]
            else:
                tensor_matrix_shape = (
                    ori_tensor_shape[0],
                    int_ceil_div(ori_tensor_shape[1], block_in),
                    int_ceil_div(ori_tensor_shape[2], block_reduce),
                    block_in,
                    block_reduce
                )
                lambda_expression = lambda batch, m1, k1, m0, k0: ori_tensor[batch, m1 * block_in + m0,
                                                                             k1 * block_reduce + k0]
        res = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info}
        )
        return res

    def compute_nd2Zz_gevm(self, ori_tensor, compute_params):
        """
        reshape nd2Zz by normal way and gevm mode
        input params:
            ori_tensor: the tensor format is nd
        return:
           the tensor format Zz
        """
        tensor_name = compute_params.get("tensor_name")
        block_in = compute_params.get("block_in")
        block_reduce = compute_params.get("block_reduce")
        data_flow = compute_params.get("data_flow")
        mode_info = compute_params.get("mode_info", "none")
        format_info = compute_params.get("format_info", "none")
        trans = compute_params.get("trans")
        int82int32_trans_flag = (data_flow == "int82int32") and trans
        if int82int32_trans_flag:
            return self._compute_nd2Zz_int8_trans(ori_tensor, compute_params)
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_tensor_shape) == 2:
            if trans:
                tensor_matrix_shape = (
                    (ori_tensor_shape[1] + block_in - 1) // block_in,
                    (ori_tensor_shape[0] + block_reduce - 1) // block_reduce,
                    block_in,
                    block_reduce
                )
                lambda_expression = lambda i, j, k, l: ori_tensor[j * block_reduce + l, i * block_in + k]
            else:
                tensor_matrix_shape = (
                    (ori_tensor_shape[0] + block_in - 1) // block_in,
                    (ori_tensor_shape[1] + block_reduce - 1) // block_reduce,
                    block_in,
                    block_reduce
                )
                lambda_expression = lambda i, j, k, l: ori_tensor[i * block_in + k, j * block_reduce + l]
        else:
            if trans:
                tensor_matrix_shape = (
                    ori_tensor_shape[0],
                    (ori_tensor_shape[2] + block_in - 1) // block_in,
                    (ori_tensor_shape[1] + block_reduce - 1) // block_reduce,
                    block_in,
                    block_reduce
                )
                lambda_expression = lambda batch, i, j, k, l: ori_tensor[batch, j * block_reduce + l, i * block_in + k]
            else:
                tensor_matrix_shape = (
                    ori_tensor_shape[0],
                    (ori_tensor_shape[1] + block_in - 1) // block_in,
                    (ori_tensor_shape[2] + block_reduce - 1) // block_reduce,
                    block_in,
                    block_reduce
                )
                lambda_expression = lambda batch, i, j, k, l: ori_tensor[batch, i * block_in + k, j * block_reduce + l]
        res_fract = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info}
        )
        if len(ori_tensor_shape) == 2:
            res_matrix = tvm.compute(
                tensor_matrix_shape,
                lambda i, j, k, l: res_fract[0, j, 0, l],
                name="tensor_a_matrix",
                attrs={"mode": mode_info, "format_info": format_info}
            )
        else:
            res_matrix = tvm.compute(
                tensor_matrix_shape,
                lambda batch, i, j, k, l: res_fract[batch, 0, j, 0, l],
                name="tensor_a_matrix",
                attrs={"mode": mode_info, "format_info": format_info}
            )
        return res_matrix

    def compute_nd2Zn(self, ori_tensor, compute_params):
        """
        reshape nd2Zn by normal way
        input params:
            ori_tensor: the tensor format is nd
        return:
           the tensor format Zn
        """
        tensor_name = compute_params.get("tensor_name")
        block_out = compute_params.get("block_out")
        block_reduce = compute_params.get("block_reduce")
        mode_info = compute_params.get("mode_info")
        format_info = compute_params.get("format_info")
        trans = compute_params.get("trans")
        data_flow = compute_params.get("data_flow")
        int82int32_no_trans_flag = (data_flow == "int82int32") and (not trans)
        if int82int32_no_trans_flag:
            return self._compute_nd2Zn_int8_no_trans(ori_tensor, compute_params)

        if trans:
            fract_shape, lambda_expression = self._lambda_nd2Nz(ori_tensor, block_out, block_reduce)
        else:
            fract_shape, lambda_expression = self._lambda_nd2Zn(ori_tensor, block_reduce, block_out)

        res = tvm.compute(
            fract_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info}
        )

        return res

    def _compute_nd2Zn_int8_no_trans(self, ori_tensor, compute_params):
        """
        this func is reshape nd2Zn when dtype is int8 and not trans and ori_shape is [k, n]
        input params:
            ori_tensor: the tensor format is nd
            compute_params: dict, the info need trans to schedule
        return:
            the tensor format is FRACTAL_Z
        """
        tensor_name = compute_params.get("tensor_name")
        block_out = compute_params.get("block_out")
        block_reduce = compute_params.get("block_reduce")
        mode_info = compute_params.get("mode_info")
        format_info = compute_params.get("format_info")

        normalize_shape = [self._get_value(i) for i in ori_tensor.shape]
        normalize_shape[-1], normalize_shape[-2] = normalize_shape[-2], normalize_shape[-1]
        if len(normalize_shape) == 2:
            tensor_fract_shape = (
                normalize_shape[1] // block_reduce,
                normalize_shape[0] // block_out,
                block_out,
                block_reduce
            )
            tensor_transpose = tvm.compute(
                normalize_shape,
                lambda i, j: ori_tensor[j, i],
                name="b_transpose"
            )
            res = tvm.compute(
                tensor_fract_shape,
                lambda i, j, k, l: tensor_transpose[j * block_out + k, i * block_reduce + l],
                name="tensor_b_matrix",
                attrs={"mode": mode_info, "format_info": format_info}
            )
        else:
            tensor_fract_shape = (
                normalize_shape[0],
                normalize_shape[2] // block_reduce,
                normalize_shape[1] // block_out,
                block_out,
                block_reduce
            )
            tensor_transpose = tvm.compute(
                normalize_shape,
                lambda batch, i, j: ori_tensor[batch, j, i],
                name="b_transpose"
            )
            res = tvm.compute(
                tensor_fract_shape,
                lambda batch, i, j, k, l: tensor_transpose[batch, j * block_out + k, i * block_reduce + l],
                name="tensor_b_matrix",
                attrs={"mode": mode_info, "format_info": format_info}
            )

        return res

    def compute_nd2Zn_vnchwconv(self, ori_tensor, compute_params):
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
        mode_info = compute_params.get("mode_info")
        format_info = compute_params.get("format_info")
        trans = compute_params.get("trans")
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if len(ori_tensor_shape) == 2:
            tensor_fract_shape = (
                ori_tensor_shape[0] // block_reduce,
                ori_tensor_shape[1],
                block_reduce
            )
            tensor_fract = tvm.compute(
                tensor_fract_shape,
                lambda i, j, k: ori_tensor[i * block_reduce + k, j],
                name="{}_fract".format(tensor_name)
            )
            if trans:
                tensor_matrix_shape = (
                    tensor_fract_shape[1] // block_out,
                    tensor_fract_shape[0],
                    block_reduce,
                    block_out
                )
                lambda_expression = lambda i, j, k, l: tensor_fract[j, i * block_reduce + l, k]
            else:
                tensor_matrix_shape = (
                    tensor_fract_shape[0],
                    tensor_fract_shape[1] // block_out,
                    block_out,
                    block_reduce
                )
                lambda_expression = lambda i, j, k, l: tensor_fract[i, j * block_out + k, l]
        else:
            tensor_fract_shape = (
                ori_tensor_shape[0],
                ori_tensor_shape[1] // block_reduce,
                ori_tensor_shape[2],
                block_reduce
            )
            tensor_fract = tvm.compute(
                tensor_fract_shape,
                lambda batch, i, j, k: ori_tensor[batch, i * block_reduce + k, j],
                name="{}_fract".format(tensor_name)
            )

            if trans:
                tensor_matrix_shape = (
                    tensor_fract_shape[0],
                    tensor_fract_shape[2] // block_out,
                    tensor_fract_shape[1],
                    block_reduce,
                    block_out
                )
                lambda_expression = lambda batch, i, j, k, l: tensor_fract[batch, j, i * block_reduce + l, k]
            else:
                tensor_matrix_shape = (
                    tensor_fract_shape[0],
                    tensor_fract_shape[1],
                    tensor_fract_shape[2] // block_out,
                    block_out,
                    block_reduce
                )
                lambda_expression = lambda batch, i, j, k, l: tensor_fract[batch, i, j * block_out + k, l]
        tensor_matrix = tvm.compute(
            tensor_matrix_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": "nd2Zn_vnchwconv", "format_info": format_info}
        )
        return tensor_matrix

    def compute_nd2Nz(self, ori_tensor, compute_params):
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
        format_info = compute_params.get("format_info")
        if not trans:
            fract_shape, lambda_expression = self._lambda_nd2Nz(ori_tensor, block_in, block_reduce)
        else:
            fract_shape, lambda_expression = self._lambda_nd2Zn(ori_tensor, block_in, block_reduce)

        res = tvm.compute(
            fract_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info}
        )

        return res

    def compute_Nz2Zz_int82fp32(self, ori_tensor, compute_params):
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
        format_info = compute_params.get("format_info")
        ori_shape = [self._get_value(i) for i in ori_tensor.shape]
        if not trans:
            if len(ori_shape) == 4:
                tensor_a_normalize_shape = [
                    ori_shape[1],
                    ori_shape[0] * 2,
                    ori_shape[2],
                    ori_shape[3] // 2
                ]
                lambda_expression = lambda i, j, k, l: ori_tensor[j // 2, i, k, (j * 16 + l) % 32]
            else:
                tensor_a_normalize_shape = [
                    ori_shape[0],
                    ori_shape[2],
                    ori_shape[1] * 2,
                    ori_shape[3],
                    ori_shape[4] // 2
                ]
                lambda_expression = lambda batch, i, j, k, l: ori_tensor[batch, j // 2, i, k, (j * 16 + l) % 32]
        else:
            if len(ori_shape) == 4:
                tensor_a_normalize_shape = [
                    ori_shape[0] * 2,
                    ori_shape[1],
                    ori_shape[2],
                    ori_shape[3] // 2
                ]
                lambda_expression = lambda i, j, k, l: ori_tensor[i // 2, j, k, (i * 16 + l) % 32]
            else:
                tensor_a_normalize_shape = [
                    ori_shape[0],
                    ori_shape[1] * 2,
                    ori_shape[2],
                    ori_shape[3],
                    ori_shape[4] // 2
                ]
                lambda_expression = lambda batch, i, j, k, l: ori_tensor[batch, i // 2, j, k, (i * 16 + l) % 32]

        res = tvm.compute(
            tensor_a_normalize_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info}
        )

        return res

    def compute_Zn2Zn_int82fp32(self, ori_tensor, compute_params):
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
        format_info = compute_params.get("format_info")
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        if not trans:
            if len(ori_tensor_shape) == 4:
                tensor_normalize_shape = (
                    ori_tensor_shape[0] * 2,
                    ori_tensor_shape[1],
                    ori_tensor_shape[2],
                    ori_tensor_shape[3] // 2
                )
                lambda_expression = lambda i, j, k, l: ori_tensor(i // 2, j, k, (i * 16 + l) % 32)
            else:
                tensor_normalize_shape = (
                    ori_tensor_shape[0],
                    ori_tensor_shape[1] * 2,
                    ori_tensor_shape[2],
                    ori_tensor_shape[3],
                    ori_tensor_shape[4] // 2
                )
                lambda_expression = lambda batch, i, j, k, l: ori_tensor(batch, i // 2, j, k, (i * 16 + l) % 32)
        else:
            if len(ori_tensor_shape) == 4:
                tensor_normalize_shape = (
                    ori_tensor_shape[1],
                    ori_tensor_shape[0] * 2,
                    ori_tensor_shape[3] // 2,
                    ori_tensor_shape[2]
                )
                lambda_expression = lambda i, j, k, l: ori_tensor(j // 2, i, l, (j * 16 + k) % 32)
            else:
                tensor_normalize_shape = (
                    ori_tensor_shape[0],
                    ori_tensor_shape[2],
                    ori_tensor_shape[1] * 2,
                    ori_tensor_shape[4] // 2,
                    ori_tensor_shape[3]
                )
                lambda_expression = lambda batch, i, j, k, l: ori_tensor(batch, j // 2, i, l, (j * 16 + k) % 32)

        res = tvm.compute(
            tensor_normalize_shape,
            lambda_expression,
            name=tensor_name,
            attrs={"mode": mode_info, "format_info": format_info}
        )

        return res

    def compute_Nz2nd(self, ori_tensor, output_shape=None, tensor_name="nz_to_nd", res_tag="", attrs_dict=None):
        """
        reshape the ori_tensor Nz to nd
        input params:
            ori_tensor: the tensor format is Nz
            tensor_name: the tensor's name after reshape in IR
        return:
            the format is ND tensor
        """
        ori_tensor_shape = [self._get_value(i) for i in ori_tensor.shape]
        block_out = ori_tensor_shape[-1]
        block_in = ori_tensor_shape[-2]

        if len(ori_tensor_shape) == 5:
            if in_dynamic():
                shape_trans = (ori_tensor_shape[0], ori_tensor_shape[2], block_in, ori_tensor_shape[1], block_out)
                before_c_gm = tvm.compute(shape_trans, lambda batch, m1, m0, n1, n0:
                    ori_tensor[batch, n1, m1, m0, n0], name="before_c_gm")
                shapes = (ori_tensor_shape[0], ori_tensor_shape[2] * block_in, ori_tensor_shape[1] * block_out)
                lambda_expression = lambda batch, m, n: before_c_gm[
                    batch,
                    m // block_in,
                    m % block_in,
                    n // block_out,
                    n % block_out
                ]
            else:
                shapes = (ori_tensor_shape[0], ori_tensor_shape[2] * block_in, ori_tensor_shape[1] * block_out)
                lambda_expression = lambda batch, m, n: ori_tensor[
                    batch,
                    n // block_out,
                    m // block_in,
                    m % block_in,
                    n % block_out
                ]
        else:
            if in_dynamic():
                shape_trans = (ori_tensor_shape[1], block_in, ori_tensor_shape[0], block_out)
                before_c_gm = tvm.compute(shape_trans, lambda m1, m0, n1, n0:
                    ori_tensor[n1, m1, m0, n0], name="before_c_gm")
                shapes = (ori_tensor_shape[1] * block_in, ori_tensor_shape[0] * block_out)
                lambda_expression = lambda m, n: before_c_gm[
                    m // block_in,
                    m % block_in,
                    n // block_out,
                    n % block_out
                ]
            else:
                shapes = (ori_tensor_shape[1] * block_in, ori_tensor_shape[0] * block_out)
                lambda_expression = lambda m, n: ori_tensor[
                    n // block_out,
                    m // block_in,
                    m % block_in,
                    n % block_out
                ]
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
