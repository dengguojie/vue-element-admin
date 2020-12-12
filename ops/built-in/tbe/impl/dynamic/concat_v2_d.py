#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

concat_v2_d: Concatenates tensors along one dimension.
            The number of dimensions of input tensors must match,
            and all dimensions except 'axis' must be equal.
            tf ConcactV2 op

"""
from __future__ import absolute_import
import math

import te.lang.dynamic
from te.utils import para_check
from te import platform as tbe_platform
from te import tik
from te.utils.error_manager import error_manager_vector as error_manager

from impl.util.util_tik_comm_func import gm2ub
from impl.util.util_tik_comm_func import ub2gm
from impl.util.util_tik_comm_func import ceil_div
from impl import common_util
from impl import constant_util as constant
MAX_SIZE = 2 ** 31 - 1


def ceil_32bytes_align_count(count, dtype):
    """
    ceil_32bytes_align_count
    """
    type_size = common_util.get_data_size(dtype)
    block_elements = constant.BLOCK_SIZE // type_size
    block_count = math.ceil(count / block_elements)
    return block_count * block_elements


# pylint: disable=too-many-arguments,too-many-locals
def _get_mask2concat_ub(instance: tik.Tik, count, src_index, dtype):
    """
    get 128bit mask for concat ub
    :param instance: tik instance
    :param count: count of element to concat from src ub to dst ub
    :param src_index: the index to concat
    :param dtype: dtype
    :return: [h_64_mask, l_64_mask]
    """
    dtype_size = common_util.get_data_size(dtype)
    if not tbe_platform.cce_conf.api_check_support("tik.vadds", dtype):
        ori_dtype_size = common_util.get_data_size(dtype)
        covert_dtype_map = {
            1: "float16",
            2: "float16",
            4: "float32",
            8: "float32",
        }

        dtype_size = common_util.get_data_size(covert_dtype_map[ori_dtype_size])
        count = count * ori_dtype_size // dtype_size
        src_index = ceil_div(instance, src_index * ori_dtype_size, dtype_size)

    # {dtype_size: (max_hight_mask, max_low_mask)}
    dtype_vadds_mask_map = {
        2: (2 ** 64 - 1, 2 ** 64 - 1),
        4: (0, 2 ** 64 - 1)
    }

    block_element = constant.BLOCK_SIZE // dtype_size
    dst_reserve = src_index % block_element
    repeat_times = count // (block_element * 8)
    if isinstance(count, int) and isinstance(src_index, int):
        h_64_mask = dtype_vadds_mask_map[dtype_size][0]
        l_64_mask = dtype_vadds_mask_map[dtype_size][1]
        l_64_mask = l_64_mask & (l_64_mask << dst_reserve)
        if repeat_times == 0 and count != block_element * 8:
            if count > 64:
                h_64_mask = h_64_mask & ((1 << (count - 64)) - 1)
            else:
                h_64_mask = 0
                l_64_mask = l_64_mask & ((1 << count) - 1)
        return [h_64_mask, l_64_mask]

    h_64_mask = instance.Scalar(dtype="int64", name="h_64_mask", init_value=dtype_vadds_mask_map[dtype_size][0])
    l_64_mask = instance.Scalar(dtype="int64", name="l_64_mask", init_value=dtype_vadds_mask_map[dtype_size][1])
    scalar_one = instance.Scalar(dtype="int64", name="scalar_one", init_value=1)
    l_64_mask.set_as(l_64_mask & (l_64_mask << dst_reserve))
    with instance.if_scope(repeat_times == 0):
        with instance.if_scope(count != block_element * 8):
            with instance.if_scope(count > 64):
                h_64_mask.set_as(h_64_mask & ((scalar_one << (count - 64)) - 1))
            with instance.else_scope():
                h_64_mask.set_as(0)
                l_64_mask.set_as(l_64_mask & ((scalar_one << count) - 1))
    return [h_64_mask, l_64_mask]


def _concat_ub_vadds(instance: tik.Tik, dst: tik.Tensor, src: tik.Tensor, dst_index, src_index, count, row_count,
                     dst_row_stride, src_row_stride, mask=None, repeat_times=None, tail_count=None):
    """
    _concat_ub_vadds
    """
    if dst.scope != tik.scope_ubuf or src.scope != tik.scope_ubuf:
        raise RuntimeError("dst and src must be UB, but dst is {} and src is {}.".format(dst.scope, src.scope))

    if dst.dtype != src.dtype:
        raise RuntimeError("dst.dtype[{}] != src.dtype[{}].".format(dst.dtype, src.dtype))

    supported_dtypes = {"float16", "float32", "int32", "int8", "int16", "int64", "uint8", "uint16", "uint32", "uint64"}
    if dst.dtype not in supported_dtypes:
        raise RuntimeError("{} is not supported, supported dtypes: {}.".format(dst.dtype, supported_dtypes))

    ori_dtype_size = common_util.get_data_size(dst.dtype)
    if not tbe_platform.cce_conf.api_check_support("tik.vadds", dst.dtype) or ori_dtype_size not in (2, 4):
        covert_dtype_map = {
            1: "float16",
            2: "float16",
            4: "float32",
            8: "float32",
        }

        if ori_dtype_size == 1:
            with instance.if_scope(dst_index % 2 != 0):
                dst[dst_index + 1] = src[src_index + 1]
        dst = dst.reinterpret_cast_to(covert_dtype_map[ori_dtype_size])
        src = src.reinterpret_cast_to(covert_dtype_map[ori_dtype_size])
        dtype_size = common_util.get_data_size(dst.dtype)
        count = count * ori_dtype_size // dtype_size
        dst_index = ceil_div(instance, dst_index * ori_dtype_size, dtype_size)
        src_index = ceil_div(instance, src_index * ori_dtype_size, dtype_size)

    dtype_size = common_util.get_data_size(dst.dtype)
    block_element = constant.BLOCK_SIZE // dtype_size

    dst_reserve = dst_index % block_element
    new_dst_index = dst_index - dst_reserve
    new_src_index = src_index - dst_reserve
    count = count + dst_reserve
    if not mask:
        mask = _get_mask2concat_ub(instance, count, src_index, dst.dtype)
    if not repeat_times:
        repeat_times = count // (block_element * 8)
    if not tail_count:
        tail_count = count - repeat_times * block_element * 8

    instance.vadds(mask, dst[new_dst_index], src[new_src_index], 0, row_count, 1, 1, dst_row_stride, src_row_stride)
    with instance.if_scope(repeat_times > 0):
        with instance.for_range(1, repeat_times) as repeat_idx:
            instance.vadds(block_element * 8, dst[new_dst_index + block_element * 8 * repeat_idx],
                           src[new_src_index + block_element * 8 * repeat_idx], 0,
                           row_count, 1, 1, dst_row_stride, src_row_stride)

        new_dst_index = new_dst_index + block_element * 8 * repeat_times
        new_src_index = new_src_index + block_element * 8 * repeat_times
        with instance.if_scope(tail_count > 0):
            instance.vadds(tail_count, dst[new_dst_index], src[new_src_index], 0, row_count, 1, 1, dst_row_stride,
                           src_row_stride)


# pylint:disable=too-many-instance-attributes,too-few-public-methods
class ConcatV2:
    """
    ConcatV2
    """
    class TilingParam:
        """
        TilingParam
        """
        def __init__(self, input_values, inst: tik.Tik):
            self.tik_instance = inst
            dtype = "int64"

            # data in tiling_gm likes:
            # 0---- 1----    2----          3----
            # axis, out_dim, max_inner_dim, min_inner_dim,
            # 4----                5----
            # output_inner_length, input_count
            # 6----    7----
            # reserve, reserve
            # 8----             9----
            # first_inner_dims, first_output_idx,
            # second_inner_dims, second_output_idx
            # ...
            self.dtype = dtype
            self.input_values = input_values
            self.axis = inst.Scalar(dtype, name="axis")
            self.out_dim = inst.Scalar(dtype, name="out_dim")
            self.max_inner_dim = inst.Scalar(dtype, name="max_inner_dim")
            self.min_inner_dim = inst.Scalar(dtype, name="min_inner_dim")
            self.output_inner_length = inst.Scalar(dtype,
                                                   name="output_inner_length")

            tiling_ub_size = len(input_values) * 2 + 8
            tiling_ub_size = ceil_32bytes_align_count(tiling_ub_size, dtype)
            tiling_gm_size = tiling_ub_size
            self.tiling_ub_size = tiling_ub_size
            self.tiling_gm = inst.Tensor(dtype, (tiling_gm_size,),
                                         name="tiling_gm",
                                         scope=tik.scope_gm)

            self._need_ub_size = (self.tiling_ub_size *
                                  common_util.get_data_size(dtype))
            self.data_dtype = self.input_values[0].get("dtype")
            self.block_element = constant.BLOCK_SIZE // common_util.get_data_size(self.data_dtype)
            self._tiling_ub = None
            self._dims = []

        def init(self):
            """
            :return:
            """
            inst = self.tik_instance
            dtype = self.dtype
            head_count = 8
            for i, _ in enumerate(self.input_values):
                self._dims.append(inst.Scalar(dtype=dtype, name="inner_dim" + str(i)))
                self._dims.append(inst.Scalar(dtype=dtype, name="output_index" + str(i)))
            with inst.new_stmt_scope():
                self._tiling_ub = inst.Tensor(dtype, (self.tiling_ub_size,),
                                              name="tiling_ub",
                                              scope=tik.scope_ubuf)
                gm2ub(inst, self._tiling_ub, self.tiling_gm, self.tiling_ub_size)
                self.axis.set_as(self._tiling_ub[0])
                self.out_dim.set_as(self._tiling_ub[1])
                self.max_inner_dim.set_as(self._tiling_ub[2])
                self.min_inner_dim.set_as(self._tiling_ub[3])
                self.output_inner_length.set_as(self._tiling_ub[4])

                for i, _ in enumerate(self.input_values):
                    index = head_count + i * 2
                    self._dims[i * 2].set_as(self._tiling_ub[index])
                    self._dims[i * 2 + 1].set_as(self._tiling_ub[index + 1])

        def get_dims(self, input_index):
            """
            :param input_index: index of input tensors
            :return: inner dims, output_index of each row
            """
            index = input_index * 2
            return self._dims[index], self._dims[index + 1]

        def update_tiling(self, multi_times):
            """
            update inner dims information multiply by multi_times
            :param multi_times: multi_times
            :return: None
            """
            self.max_inner_dim.set_as(self.max_inner_dim * multi_times)
            self.min_inner_dim.set_as(self.min_inner_dim * multi_times)
            self.output_inner_length.set_as(self.output_inner_length * multi_times)
            for _, dim_i in enumerate(self._dims):
                dim_i.set_as(dim_i * multi_times)

        # pylint: disable=no-self-use
        def need_ub_size(self):
            """
            :return: 0
            """
            return 0

    def __init__(self, input_values, axis, kernel_name):
        self.tik_instance = tik.Tik()
        self.tik_profiling = tik.Dprofile()
        self.tiling_param = self.TilingParam(input_values, self.tik_instance)
        self.aicore_num = self.tik_profiling.get_aicore_num()
        self.kernel_name = kernel_name
        self.axis = axis

        self.dtype = input_values[0].get("dtype").lower()
        self.output_shape = (MAX_SIZE,)
        self.input_shape = (MAX_SIZE,)

        self.input_tensors, self.output_tensor = self._init_gm_tensor(self.input_shape, self.output_shape,
                                                                      len(input_values),
                                                                      self.dtype)

        dtype_bytes_size = common_util.get_data_size(self.dtype)
        self.type_size = dtype_bytes_size
        self.ele_each_block = constant.BLOCK_SIZE // dtype_bytes_size
        valid_ub_size = self.tik_profiling.get_unified_buffer_size()
        valid_ub_size -= self.tiling_param.need_ub_size()
        self.ub_buffer_length = valid_ub_size

        # reserve two block size for not 32 bytes align
        self.ub_buffer_length -= constant.BLOCK_SIZE * 2

        # make ub_buffer_length 32 bytes align
        self.ub_buffer_length //= constant.BLOCK_SIZE
        self.ub_buffer_length *= constant.BLOCK_SIZE

        self.ub_buffer_length //= dtype_bytes_size

    def _init_gm_tensor(self, input_shape, output_shape, input_count, dtype):
        """
        init gm tensor

        Parameters
        ----------
        input_shape: list
            shape of input tensor
        output_shape: list
            shape of output tensor
        dtype: str
            data type

        Returns
        -------
        input_tensors: tik tensor
            input gm tensor
        output_tensor: tik tensor
            output gm tensor
        """
        input_tensors = []
        for _, index in enumerate(range(input_count)):
            tensor_name = "gm_input_" + str(index)
            gm_tensor = self.tik_instance.Tensor(dtype, input_shape, name=tensor_name, scope=tik.scope_gm)
            input_tensors.append(gm_tensor)

        output_tensor = self.tik_instance.Tensor(dtype, output_shape, name="gm_output", scope=tik.scope_gm)

        return input_tensors, output_tensor

    def concat_compute(self):
        """
        build concat op

        Returns
        -------
        None
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        with inst.for_range(0, aicore_num, name="core_idx", block_num=aicore_num) as i:
            self.tiling_param.init()
            min_inner_dim = self.tiling_param.min_inner_dim
            with inst.if_scope(min_inner_dim < self.ele_each_block):
                self._concat_small_inner(i)
            with inst.else_scope():
                self._concat_large_inner(i)

        opt_config = {"out_of_bound_sync_check": True,
                      "enable_const_fold": True}
        inst.BuildCCE(kernel_name=self.kernel_name, inputs=self.input_tensors, outputs=(self.output_tensor,),
                      flowtable=[self.tiling_param.tiling_gm],
                      config=opt_config,
                      enable_l2=False)

        te.op.add_compile_info("vars", {"input_size": len(self.input_tensors),
                                        "concat_dim": self.axis,
                                        "block_dim": self.aicore_num
                                        })
        return inst

    def _get_ceil_32bytes_count(self, count: tik.Scalar):
        ceil_num = ceil_div(self.tik_instance, count, self.ele_each_block)
        return ceil_num * self.ele_each_block

    # pylint: disable=invalid-name,unused-variable,too-many-statements
    def _concat_inner_dim_each_split(self, out_dim_idx, inner_dim_split_idx):
        for index, _ in enumerate(self.input_tensors):
            self._concat_compute_tensor_inner_dim(out_dim_idx, inner_dim_split_idx, index)

    def _concat_compute_tensor_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        with self.tik_instance.if_scope(inner_dims % self.ele_each_block == 0):
            self._concat_tensor_align_inner_dim(out_dim_idx, inner_dim_split_idx, tensor_index)
        with self.tik_instance.else_scope():
            self._concat_tensor_not_align_inner_dim(out_dim_idx, inner_dim_split_idx, tensor_index)

    def _concat_tensor_align_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        inst = self.tik_instance
        factor = self.ub_buffer_length
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        input_gm = self.input_tensors[tensor_index]
        output_gm = self.output_tensor
        with inst.new_stmt_scope():
            ub_length = self.ub_buffer_length
            ub = inst.Tensor(input_gm.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            in_start_index = inner_dim_split_idx * factor + inner_dims * out_dim_idx

            output_dim = self.tiling_param.output_inner_length
            out_start_index = output_idx + inner_dim_split_idx * factor + output_dim * out_dim_idx
            with inst.if_scope(in_start_index < inner_dims * (1 + out_dim_idx)):
                count = inst.Scalar("int64", name="count")
                count.set_as(inner_dims * (1 + out_dim_idx) - in_start_index)
                with inst.if_scope(count > ub_length):
                    count.set_as(ub_length)

                gm2ub(inst, ub, input_gm[in_start_index:], count)
                ub2gm(inst, output_gm[out_start_index:], ub, count)

    def _concat_tensor_not_align_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        inst = self.tik_instance
        factor = self.ub_buffer_length
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        input_gm = self.input_tensors[tensor_index]
        output_gm = self.output_tensor

        with inst.new_stmt_scope():
            ub_length = self.ub_buffer_length
            ub = inst.Tensor(input_gm.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            in_start_index = inner_dim_split_idx * factor + inner_dims * out_dim_idx

            output_dim = self.tiling_param.output_inner_length
            out_start_index = output_idx + inner_dim_split_idx * factor + output_dim * out_dim_idx
            with inst.if_scope(in_start_index < inner_dims * (1 + out_dim_idx)):
                count = inner_dims * (1 + out_dim_idx) - in_start_index
                with inst.if_scope(count > ub_length):
                    gm2ub(inst, ub, input_gm[in_start_index:], ub_length)
                    ub2gm(inst, output_gm[out_start_index:], ub, ub_length)
                with inst.else_scope():
                    with inst.if_scope(inner_dim_split_idx > 0):
                        align_count = self._get_ceil_32bytes_count(count)
                        redundant_count = align_count - count
                        new_in_start_index = in_start_index - redundant_count
                        new_out_start_index = out_start_index - redundant_count
                        gm2ub(inst, ub, input_gm[new_in_start_index:], align_count)
                        ub2gm(inst, output_gm[new_out_start_index:], ub, align_count)
                    with inst.else_scope():
                        gm2ub(inst, ub, input_gm[in_start_index:], self.ele_each_block)
                        ub2gm(inst, output_gm[out_start_index:], ub, self.ele_each_block)

                        in_start_index += self.ele_each_block
                        out_start_index += self.ele_each_block
                        align_count = self._get_ceil_32bytes_count(count - self.ele_each_block)
                        redundant_count = align_count - count + self.ele_each_block
                        new_in_start_index = in_start_index - redundant_count
                        new_out_start_index = out_start_index - redundant_count
                        gm2ub(inst, ub, input_gm[new_in_start_index:], align_count)
                        ub2gm(inst, output_gm[new_out_start_index:], ub, align_count)

    def _concat_large_inner(self, core_idx):
        """
        tiling with out_dims and split of inner_dims
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        out_dims = self.tiling_param.out_dim
        max_inner_dim = self.tiling_param.max_inner_dim
        inner_dims_loops = ceil_div(inst, max_inner_dim, self.ub_buffer_length)
        max_loops = out_dims * inner_dims_loops

        out_loops = ceil_div(inst, max_loops, aicore_num)
        with inst.for_range(0, out_loops, name="out_loops_idx") as i:
            loop_idx = i + out_loops * core_idx
            with inst.if_scope(loop_idx < max_loops):
                out_dim_idx = loop_idx / inner_dims_loops
                inner_dim_split_idx = loop_idx % inner_dims_loops
                self._concat_inner_dim_each_split(out_dim_idx, inner_dim_split_idx)

    def _concat_small_inner(self, core_idx):
        """
        tiling with out_dims
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        out_dims = self.tiling_param.out_dim
        ub_len = (self.ub_buffer_length // 4) // self.ele_each_block * self.ele_each_block
        output_inner_dim = self.tiling_param.output_inner_length
        ub_can_storage_lines_vnchwconv = ub_len // self.ele_each_block
        ub_can_copy_lines = inst.Scalar(dtype="int64", name="ub_can_copy_lines",
                                        init_value=self.ub_buffer_length // (self.tiling_param.output_inner_length * 4))
        lines_each_core = ceil_div(inst, ceil_div(inst, out_dims, self.aicore_num),
                                   self.ele_each_block) * self.ele_each_block

        with inst.if_scope(tik.all(output_inner_dim >= self.ele_each_block,
                                   ub_can_copy_lines >= self.ele_each_block,
                                   output_inner_dim < 256)):
            with inst.if_scope(lines_each_core < ub_can_copy_lines):
                ub_can_copy_lines.set_as(lines_each_core)
            self._concat_small_inner_each_core_multi_line(core_idx, out_dims, ub_can_copy_lines)
        with inst.else_scope():
            count_each_core = ceil_div(inst, out_dims, aicore_num)
            self._concat_small_inner_each_core_one_line(core_idx, out_dims, count_each_core)

    def _concat_small_inner_each_core_one_line(self, core_idx, out_dims, count_each_core):
        inst = self.tik_instance
        with inst.for_range(0, count_each_core, name="inner_loop") as j:
            row_idx = j + count_each_core * core_idx
            with inst.if_scope(row_idx < out_dims):
                with inst.if_scope(j != count_each_core - 1):
                    self._concat_small_inner_each_core_not_last_row(row_idx)
                with inst.else_scope():
                    self._concat_small_inner_each_core_last_row(row_idx)

    def _concat_small_inner_each_core_not_last_row(self, row_idx):
        self._concat_small_inner_each_core_without_treat_overlap(row_idx, self.input_tensors)

    def _concat_small_inner_each_core_last_row(self, row_idx):
        self._concat_small_inner_each_core_without_treat_overlap(row_idx,
                                                                 self.input_tensors[0:len(self.input_tensors) - 1])
        self._concat_small_inner_each_core_last_row_last_tensor(row_idx)

    def _concat_small_inner_each_core_without_treat_overlap(self, row_idx, tensors):
        inst = self.tik_instance
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        ub_length = self.ub_buffer_length
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            ub_data_count = inst.Scalar("int32", name="ub_data_count")
            ub_data_count.set_as(0)
            tmp_ub = inst.Tensor(self.dtype, (self.ele_each_block,), scope=tik.scope_ubuf, name="tmp_ub")

            out_row_start_idx = output_inner_len * row_idx
            out_start_idx = inst.Scalar("int64", name="ub_data_count")
            out_start_idx.set_as(out_row_start_idx)
            for index, input_tensor in enumerate(tensors):
                inner_dim, output_idx = self.tiling_param.get_dims(index)
                in_start_idx = inner_dim * row_idx
                with inst.if_scope(ub_data_count >= self.ele_each_block):
                    ub2gm(inst, output_tensor[out_start_idx:], out_ub, ub_data_count)
                    ub_data_count.set_as(0)

                with inst.if_scope(ub_data_count == 0):
                    out_start_idx.set_as(out_row_start_idx + output_idx)

                with inst.if_scope(inner_dim < self.ele_each_block):
                    gm2ub(inst, tmp_ub, input_tensor[in_start_idx:], inner_dim)
                    with inst.for_range(0, inner_dim) as scalar_idx:
                        out_ub[ub_data_count] = tmp_ub[scalar_idx]
                        ub_data_count.set_as(ub_data_count + 1)

                with inst.else_scope():
                    with inst.if_scope(ub_data_count > 0):
                        ub2gm(inst, output_tensor[out_start_idx:], out_ub, ub_data_count)
                        ub_data_count.set_as(0)
                        out_start_idx.set_as(out_row_start_idx + output_idx)

                    loops = ceil_div(inst, inner_dim, ub_length)
                    with inst.for_range(0, loops, name="inner_loop") as idx:
                        in_start_idx = ub_length * idx + inner_dim * row_idx
                        out_start_idx.set_as(ub_length * idx + out_row_start_idx + output_idx)
                        count = inst.Scalar("int64", name="count")
                        count.set_as(inner_dim * (1 + row_idx) - in_start_idx)
                        with inst.if_scope(count > ub_length):
                            count.set_as(ub_length)

                        gm2ub(inst, out_ub, input_tensor[in_start_idx:], count)
                        ub2gm(inst, output_tensor[out_start_idx:], out_ub, count)

            with inst.if_scope(ub_data_count > 0):
                ub2gm(inst, output_tensor[out_start_idx:], out_ub, ub_data_count)

    def _concat_small_inner_each_core_last_row_last_tensor(self, row_idx):
        inst = self.tik_instance
        inst = self.tik_instance
        ub_length = self.ub_buffer_length
        output_inner_len = self.tiling_param.output_inner_length
        out_dims = self.tiling_param.out_dim
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            output_tensor = self.output_tensor
            last_idx = len(self.input_tensors) - 1
            input_tensor = self.input_tensors[last_idx]
            inner_dim, output_idx = self.tiling_param.get_dims(last_idx)
            out_start_idx = inst.Scalar("int64", name="ub_data_count")
            ub_data_count = inst.Scalar("int32", name="ub_data_count")
            tmp_ub = inst.Tensor(self.dtype, (self.ele_each_block,), scope=tik.scope_ubuf, name="tmp_ub")
            out_start_idx.set_as(row_idx * output_inner_len + output_idx)
            with inst.if_scope(inner_dim < self.ele_each_block):
                gm2ub(inst, out_ub, input_tensor[inner_dim * row_idx], inner_dim)
                ub_data_count.set_as(inner_dim)
                pad_count = inst.Scalar("int32", name="pad_count")
                pad_count.set_as(self.ele_each_block - inner_dim)
                loops = ceil_div(inst, pad_count, output_inner_len)
                with inst.for_range(0, loops) as loop:
                    new_out_dim_idx = row_idx + loop
                    with inst.if_scope(new_out_dim_idx < out_dims):
                        for idx, tmp_tensor in enumerate(self.input_tensors):
                            temp_inner_dims, _ = self.tiling_param.get_dims(idx)
                            with inst.if_scope(ub_data_count < self.ele_each_block):
                                gm2ub(inst, tmp_ub, tmp_tensor[(row_idx + loop + 1) * temp_inner_dims],
                                      self.ele_each_block)
                                with inst.for_range(0, temp_inner_dims) as scalar_idx:
                                    with inst.if_scope(ub_data_count < self.ele_each_block):
                                        out_ub[ub_data_count] = tmp_ub[scalar_idx]
                                        ub_data_count.set_as(ub_data_count + 1)

                ub2gm(inst, output_tensor[out_start_idx:], out_ub, inner_dim)
            with inst.else_scope():
                loops = ceil_div(inst, inner_dim, ub_length)
                with inst.for_range(0, loops, name="inner_loop") as idx:
                    in_start_idx = (ub_length * idx + inner_dim * row_idx)
                    out_start_idx.set_as(ub_length * idx + output_inner_len * row_idx + output_idx)
                    count = inner_dim * (row_idx + 1) - in_start_idx
                    with inst.if_scope(count > ub_length):
                        gm2ub(inst, out_ub, input_tensor[in_start_idx:], ub_length)
                        ub2gm(inst, output_tensor[out_start_idx:], out_ub, ub_length)
                    with inst.else_scope():
                        with inst.if_scope(idx > 0):
                            align_count = self._get_ceil_32bytes_count(count)
                            redundant_cnt = (align_count - count)
                            new_in_start_index = in_start_idx - redundant_cnt
                            new_out_start_index = out_start_idx - redundant_cnt
                            gm2ub(inst, out_ub, input_tensor[new_in_start_index:], count)
                            ub2gm(inst, output_tensor[new_out_start_index:], out_ub, count)
                        with inst.else_scope():
                            gm2ub(inst, out_ub, input_tensor[in_start_idx:], self.ele_each_block)
                            ub2gm(inst, output_tensor[out_start_idx:], out_ub, self.ele_each_block)
                            in_start_idx += self.ele_each_block
                            out_start_idx += self.ele_each_block
                            align_count = self._get_ceil_32bytes_count(count - self.ele_each_block)
                            redundant_cnt = align_count - count + self.ele_each_block
                            new_in_start_index = in_start_idx - redundant_cnt
                            new_out_start_index = out_start_idx - redundant_cnt
                            gm2ub(inst, out_ub, input_tensor[new_in_start_index:], align_count)
                            ub2gm(inst, output_tensor[new_out_start_index:], out_ub, align_count)

    def _concat_small_inner_each_core_multi_line(self, core_idx, out_dims, ub_can_copy_lines):
        inst = self.tik_instance
        ub_copy_times = ceil_div(inst, out_dims, ub_can_copy_lines)
        ub_copy_times_each_core = ceil_div(inst, ub_copy_times, self.aicore_num)
        to_do_count = inst.Scalar(dtype="int64", name="to_do_count")
        with inst.for_range(0, ub_copy_times_each_core, name="inner_loop", thread_num=1) as j:
            row_idx = j * ub_can_copy_lines + ub_can_copy_lines * ub_copy_times_each_core * core_idx
            with inst.if_scope(row_idx < out_dims):
                with inst.if_scope(row_idx + ub_can_copy_lines <= out_dims):
                    to_do_count.set_as(ub_can_copy_lines)
                with inst.else_scope():
                    to_do_count.set_as(out_dims - row_idx)
                if self.type_size % 2 != 1:
                    with inst.if_scope(self.tiling_param.max_inner_dim >= self.ele_each_block):
                        self._concat_small_inner_each_core_multi_line_by_vadds(row_idx, to_do_count)
                    with inst.else_scope():
                        self._concat_small_inner_each_core_multi_line_by_scalar(row_idx, to_do_count)
                else:
                    self._concat_small_inner_each_core_multi_line_by_scalar(row_idx, to_do_count)

    def _concat_small_inner_each_core_multi_line_by_vadds(self, row_idx, lines):
        inst = self.tik_instance
        tensors = self.input_tensors
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        ub_length = self.ub_buffer_length // 4
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            tmp_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="tmp_ub")
            row_idx = inst.Scalar("int64", name="row_idx", init_value=row_idx)
            lines = inst.Scalar("int64", name="lines", init_value=lines)

            with inst.if_scope(tik.all(row_idx == 0, lines >= self.ele_each_block)):
                # if core 0 first handle self.ele_each_block rows
                self._concat_small_inner_each_core_multi_lines_first_block_element_rows(out_ub, tmp_ub, lines)
                row_idx.set_as(row_idx + self.ele_each_block)
                lines.set_as(lines - self.ele_each_block)
            
            with inst.if_scope(lines > 0):
                out_row_start_idx = output_inner_len * row_idx
                out_start_idx = inst.Scalar("int64", name="ub_data_count")
                out_start_idx.set_as(out_row_start_idx)
                repeat_times = lines // self.ele_each_block
                for index, input_tensor in enumerate(tensors):
                    inner_dim, output_idx = self.tiling_param.get_dims(index)
                    with inst.if_scope(inner_dim > 0):
                        with inst.if_scope(repeat_times > 0):
                            with inst.for_range(row_idx, row_idx + self.ele_each_block) as line_idx:
                                out_ub_idx = output_inner_len * (line_idx - row_idx) + output_idx
                                pre_redundant_cnt = out_ub_idx % self.ele_each_block
                                gm2ub(inst, tmp_ub, input_tensor[inner_dim * line_idx - pre_redundant_cnt:],
                                      inner_dim * repeat_times * self.ele_each_block + pre_redundant_cnt)
                                _concat_ub_vadds(inst, out_ub, tmp_ub, out_ub_idx, pre_redundant_cnt, inner_dim,
                                                 repeat_times, output_inner_len, inner_dim)

                        with inst.for_range(row_idx + repeat_times * self.ele_each_block, row_idx + lines) as line_idx:
                            out_ub_idx = output_inner_len * (line_idx - row_idx) + output_idx
                            pre_redundant_cnt = out_ub_idx % self.ele_each_block
                            gm2ub(inst, tmp_ub, input_tensor[inner_dim * line_idx - pre_redundant_cnt:],
                                  inner_dim + pre_redundant_cnt)
                            _concat_ub_vadds(inst, out_ub, tmp_ub, out_ub_idx, pre_redundant_cnt, inner_dim, 1, 0, 0)
                ub2gm(inst, output_tensor[out_start_idx:], out_ub, output_inner_len * lines)

    def _concat_small_inner_each_core_multi_line_by_scalar(self, row_idx, lines):
        inst = self.tik_instance
        tensors = self.input_tensors
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        ub_length = self.ub_buffer_length // 4
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            tmp_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="tmp_ub")
            out_start_idx = output_inner_len * row_idx
            for index, input_tensor in enumerate(tensors):
                inner_dim, output_idx = self.tiling_param.get_dims(index)
                with inst.if_scope(inner_dim > 0):
                    gm2ub(inst, tmp_ub, input_tensor[inner_dim * row_idx:], inner_dim * lines)
                    with inst.for_range(0, lines) as line_idx:
                        with inst.for_range(0, inner_dim) as element_idx:
                            out_ub[output_inner_len * line_idx + output_idx + element_idx] = tmp_ub[inner_dim *
                                                                                                    line_idx +
                                                                                                    element_idx]
            ub2gm(inst, output_tensor[out_start_idx:], out_ub, output_inner_len * lines)

    def _concat_small_inner_each_core_multi_lines_first_block_element_rows(self, out_ub, tmp_ub, lines):
        inst = self.tik_instance
        tensors = self.input_tensors
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        ub_list = [out_ub, tmp_ub]
        with inst.for_range(0, self.ele_each_block - 1) as ii:
            for index, input_tensor in enumerate(tensors):
                inner_dim, output_idx = self.tiling_param.get_dims(index)
                cur_out_ub = ub_list[index % len(ub_list)]
                gm2ub(inst, cur_out_ub, input_tensor[inner_dim * ii], inner_dim)
                ub2gm(inst, output_tensor[output_inner_len * ii + output_idx], cur_out_ub, inner_dim)

        ii = self.ele_each_block - 1
        for index, input_tensor in enumerate(tensors):
            inner_dim, output_idx = self.tiling_param.get_dims(index)
            align_size = ceil_div(inst, inner_dim, self.ele_each_block) * self.ele_each_block
            with inst.if_scope(tik.any(lines > self.ele_each_block, output_idx + align_size <= output_inner_len)):
                burst = ceil_div(inst, inner_dim, self.ele_each_block)
                cur_out_ub = ub_list[index % len(ub_list)]
                gm2ub(inst, cur_out_ub, input_tensor[inner_dim * ii], inner_dim, burst)
                ub2gm(inst, output_tensor[output_inner_len * ii + output_idx], cur_out_ub, inner_dim, burst)
            with inst.else_scope():
                reserve_count = align_size - inner_dim
                burst = ceil_div(inst, align_size, self.ele_each_block)
                gm2ub(inst, tmp_ub, input_tensor[inner_dim * ii - reserve_count], align_size, burst)
                gm2ub(inst, out_ub, output_tensor[output_inner_len * ii + output_idx - reserve_count],
                      align_size, burst)
                _concat_ub_vadds(inst, out_ub, tmp_ub, reserve_count, reserve_count, inner_dim, 1, 0, 0)
                ub2gm(inst, output_tensor[output_inner_len * ii + output_idx - reserve_count], out_ub,
                      align_size, burst)


def _check_shape(input_values, shape_name):
    # check the length of input shape must be equal
    dim_num = len(input_values[0].get(shape_name))
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get(shape_name)
        if len(shape_input) != dim_num:
            error_manager.raise_err_check_params_rules("concat", "The length of each shape must be equal",
                                                       "input_values",
                                                       [i.get(shape_name) for i in input_values])


def __check_params(input_values, axis):
    _check_shape(input_values, "shape")
    _check_shape(input_values, "ori_shape")

    dim_num = len(input_values[0].get("ori_shape"))

    if axis >= dim_num or axis < -dim_num:
        error_manager.raise_err_input_value_invalid("concat",
                                                    "concat_dim",
                                                    "between " + str(min(-dim_num, dim_num - 1)) + " and " +
                                                    str(max(-dim_num, dim_num - 1)),
                                                    axis)

    shape_value = []
    for _, tensor_dict in enumerate(input_values):
        shape_value.append(tensor_dict.get("ori_shape"))
    first_input_shape = input_values[0].get("ori_shape")

    # dims must equal except merge axis
    axis_new = axis % dim_num
    for j, _ in enumerate(first_input_shape):
        if j == axis_new:
            continue

        dim_values = set()
        for _, element_shape in enumerate(shape_value):
            dim_values.add(element_shape[j])

        if -1 in dim_values:
            dim_values.remove(-1)

        if len(dim_values) > 1:
            error_manager.raise_err_check_params_rules("concat",
                                                       "Dims must be equal except merge concat axis[%s]" % axis,
                                                       "input_values",
                                                       shape_value)

    dtype_lists = []
    for input_value in input_values:
        input_format = input_value.get("format")
        dtype_lists.append(input_value.get("dtype"))
        supported_formats = {"ND", "NHWC", "NCHW"}
        if input_format not in supported_formats:
            error_manager.raise_err_input_format_invalid('concat',
                                                         'input_values',
                                                         ','.join(supported_formats),
                                                         input_format)

    dtype = dtype_lists[0]
    for index, dtype_ in enumerate(dtype_lists):
        if dtype != dtype_:
            error_manager.raise_err_inputs_dtype_not_equal("concat",
                                                           "input_values[0]",
                                                           "input_values[%s]" % index,
                                                           dtype,
                                                           dtype_)


# pylint: disable=unused-argument
@te.op.register_operator("ConcatV2D")
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def concat_v2_d(input_values, output_data, axis, kernel_name="concat_v2_d"):
    """
    algorithm: concat_v2_d

    Parameters
    ----------
    input_values : A list of dict objects.
                 dict with keys(shape and dtype) of tensor
                 dtype only support float32, int8, int16, int32, int64, uint8,
                 uint16, uint32, uint64, float16
    output_data : A dict resulting from concatenation of the input tensors
    axis : scalar,in the range [-rank(values), rank(values))
    kernel_name : cce kernel name, default value is "concat_v2_d"

    Returns
    -------
    tik instance
    """
    __check_params(input_values, axis)
    concat_instance = ConcatV2(input_values, axis, kernel_name)
    return concat_instance.concat_compute()
