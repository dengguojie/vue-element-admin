#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

strided slice
"""

from __future__ import absolute_import, with_statement
import math
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik

from impl import common_util
from impl import constant_util as constant
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_tik_comm_func import ceil_align, ceil_div, floor_align

MAX_SIZE = 2 ** 31 - 1
MAX_NBURST = 4095
MAX_REPEAT = 255
Tiling_UB_SIZE = 382


def check_supported(input_x, begin, end, strides=None,
                    output_x=None, begin_mask=0, end_mask=0,
                    ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                    kernel_name="strided_slice"):
    return "Unknown"


def ceil_32bytes_align_count(count, dtype):
    """
    ceil_32bytes_align_count
    """
    type_size = common_util.get_data_size(dtype)
    block_count = math.ceil(count * type_size / constant.BLOCK_SIZE)
    return block_count * constant.BLOCK_SIZE // type_size


def _data_move(tik_instance: tik.Tik, dest: tik.Tensor, src: tik.Tensor, count):
    """
    _data_move
    """
    dtype_size = common_util.get_data_size(src.dtype)
    burst = math.ceil(count * dtype_size / constant.BLOCK_SIZE)
    tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)


# pylint: disable=too-many-locals, too-many-statements, too-many-instance-attributes
# pylint: disable=too-few-public-methods
class StridedSlice:
    """
    StridedSlice
    """

    # pylint: disable=too-many-locals, too-many-statements, too-many-instance-attributes
    class TilingParam:
        """
        TilingParam
        """

        def __init__(self, input_x_shape, inst: tik.Tik):
            """
            tiling param
            :param input_x_shape: input shape
            :param inst: tik instance
            """
            self.tik_instance = inst
            dtype = "int64"
            self.dtype = dtype
            # mode_type, shape_length, input_shape, output_shape, begin, end, stride
            tiling_gm_size = 2 + len(input_x_shape) * 5
            self.tiling_gm = inst.Tensor(dtype, (tiling_gm_size,), name="tiling_gm", scope=tik.scope_gm)

            self.input_shape = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="input_dims")
            self.begin = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="begin_dims")
            self.end = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="end_dims")
            self.stride = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="stride_dims")
            self.output_shape = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="output_shape_dims")
            self.input_steps = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="input_steps")
            self.output_steps = inst.ScalarArray(dtype=dtype, length=len(input_x_shape), name="output_steps")

            self.shape_length = inst.Scalar(dtype, name="shape_length", init_value=len(input_x_shape))
            self.tiling_mode = inst.Scalar(dtype, name="tiling_mode")
            self.out_dim = inst.Scalar(dtype, name="out_dim")
            self.out_dim_with_vnchwconv = inst.Scalar(dtype, name="out_dim_with_vnchwconv")

        # pylint: disable=invalid-name
        def init(self):
            """
            init process data
            """
            with self.tik_instance.new_stmt_scope():
                need_ub_size = ceil_32bytes_align_count(self.tiling_gm.shape[0], self.dtype)
                tiling_ub = self.tik_instance.Tensor(self.dtype, (need_ub_size,), name="tiling_ub",
                                                     scope=tik.scope_ubuf)
                _data_move(self.tik_instance, tiling_ub, self.tiling_gm, need_ub_size)

                self.tiling_mode.set_as(tiling_ub[0])
                self.shape_length.set_as(tiling_ub[1])
                index = self.tik_instance.Scalar(init_value=2)
                items = (self.input_shape, self.output_shape, self.begin, self.end, self.stride)
                for item in items:
                    with self.tik_instance.for_range(0, self.shape_length) as dim_idx:
                        item[dim_idx].set_as(tiling_ub[index])
                        index.set_as(index + 1)

            self.out_dim.set_as(1)
            self.out_dim_with_vnchwconv.set_as(1)
            with self.tik_instance.for_range(0, self.shape_length) as index:
                dim = self.output_shape[index]
                with self.tik_instance.if_scope(index < self.shape_length - 1):
                    self.out_dim.set_as(self.out_dim * dim)
                with self.tik_instance.if_scope(index < self.shape_length - 2):
                    self.out_dim_with_vnchwconv.set_as(self.out_dim_with_vnchwconv * dim)

            with self.tik_instance.for_range(0, self.shape_length) as index:
                dim_idx = self.shape_length - 1 - index
                self.input_steps[index].set_as(self.input_shape[dim_idx])
                self.output_steps[index].set_as(self.output_shape[dim_idx])
                with self.tik_instance.if_scope(index > 0):
                    self.input_steps[index].set_as(self.input_steps[index] * self.input_steps[index - 1])
                    self.output_steps[index].set_as(self.output_steps[index] * self.output_steps[index - 1])

    # pylint: disable=locally-disabled,too-many-arguments,
    # pylint: disable=unused-argument,too-many-locals
    def __init__(self, input_x, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                 kernel_name="strided_slice"):
        self.strides = strides
        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.ellipsis_mask = ellipsis_mask
        self.new_axis_mask = new_axis_mask
        self.shrink_axis_mask = shrink_axis_mask
        self.kernel_name = kernel_name

        inst = tik.Tik()
        self.tik_instance = inst
        self.tik_profiling = tik.Dprofile()
        max_dim_supported = 8
        self.tiling_param = self.TilingParam([1] * max_dim_supported, inst)
        self.dtype = input_x.get("dtype").lower()
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.input_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="input_gm", scope=tik.scope_gm)
        self.begin_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="begin_gm", scope=tik.scope_gm)
        self.end_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="end_gm", scope=tik.scope_gm)
        self.strides_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="strides_gm", scope=tik.scope_gm)
        self.output_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="output_gm", scope=tik.scope_gm)
        self.aicore_num = self.tik_profiling.get_aicore_num()
        self.block_element = constant.BLOCK_SIZE // self.dtype_size
        self.reserve_ub_size = Tiling_UB_SIZE
        self.ub_size = (self.tik_profiling.get_unified_buffer_size() - self.reserve_ub_size) // self.dtype_size // \
                       self.block_element * self.block_element
        self.ub_size_with_vnchwconv = ((self.tik_profiling.get_unified_buffer_size() - self.reserve_ub_size) // \
                                       self.dtype_size - self.block_element) // 2 // \
                                       self.block_element * self.block_element
        self.max_gap = 65535 * self.block_element
        self.max_last_dim = (self.max_gap + self.ub_size) // self.block_element
        self.shape_length = None

    def _ceil_div(self, int1: tik.Scalar, int2):
        """
        get ceil for (int1 / int2)
        """
        result = self.tik_instance.Scalar("int64")
        with self.tik_instance.if_scope(int1 == 0):
            result.set_as(1)
        with self.tik_instance.else_scope():
            result.set_as(int1 // int2)
        with self.tik_instance.if_scope(int1 % int2 != 0):
            result.set_as(result + 1)

        return result

    def _ceil_32bytes_count(self, count: tik.Scalar, block_element):
        """
        _ceil_32bytes_count
        """
        ceil_num = self._ceil_div(count, block_element)
        return ceil_num * block_element

    def _get_input_gm_addr(self, cur_index: tik.Scalar):
        """
        _get_input_gm_addr
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="input_addr")
        addr.set_as(self.tiling_param.begin[dim_count - 1])
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(2, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.input_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * (tmp + self.tiling_param.begin[dim_count - dim_idx]))
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _get_output_gm_addr(self, cur_index: tik.Scalar):
        """
        _get_output_gm_addr
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="output_addr")
        addr.set_as(0)
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(2, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.output_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * tmp)
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _data_move(self, dest: tik.Tensor, src: tik.Tensor, count: tik.Scalar):
        """
        _data_move
        """
        dtype_size = common_util.get_data_size(src.dtype)
        burst = self._ceil_div(count * dtype_size, constant.BLOCK_SIZE)
        self.tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)

    def strided_slice(self):
        """
        strided_slice
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        if self.dtype == "float16":
            with inst.for_range(0, core_num, block_num=core_num, name="core_idx") as i:
                self.tiling_param.init()
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(self.tiling_param.tiling_mode == 1):
                    self._do_small_last_dim(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 2):
                    self._do_large_last_dim(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 3):
                    self._do_small_last_dim_with_vnchwconv(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 5):
                    self._do_only_slice_last_dim_with_vnchwconv(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 8):
                    self._do_with_last_dim_equal_one(i)
        elif self.dtype_size % 2 == 0:
            with inst.for_range(0, core_num, block_num=core_num, name="core_idx") as i:
                self.tiling_param.init()
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(tik.any(self.tiling_param.tiling_mode == 1, self.tiling_param.tiling_mode == 3)):
                    self._do_small_last_dim(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 2):
                    self._do_large_last_dim(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 5):
                    self._do_only_slice_last_dim_with_vnchwconv(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 8):
                    self._do_with_last_dim_equal_one(i)
        else:
            with inst.for_range(0, core_num, block_num=core_num, name="core_idx") as i:
                self.tiling_param.init()
                self.shape_length = self.tiling_param.shape_length
                with inst.if_scope(tik.any(self.tiling_param.tiling_mode == 1, self.tiling_param.tiling_mode == 3)):
                    self._do_small_last_dim(i)
                with inst.elif_scope(tik.any(self.tiling_param.tiling_mode == 2, self.tiling_param.tiling_mode == 5)):
                    self._do_large_last_dim(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 4):
                    self._do_large_last_dim_multi_rows(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 6):
                    self._do_only_slice_last_dim_with_datamove(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 7):
                    self._do_with_one_dim(i)
                with inst.elif_scope(self.tiling_param.tiling_mode == 8):
                    self._do_with_last_dim_equal_one(i)

    def _do_with_last_dim_equal_one(self, core_idx):
        """
        slice the last dim and the last dim size of outshape is equal one.
        Parameters
        ----------
        core_idx: number of ai_core

        Returns
        -------
        None
        """
        inst = self.tik_instance
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        max_rows_in_ub = inst.Scalar("int64", name="max_rows_in_ub")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        last_repeat_rows = inst.Scalar("int64", name="last_repeat_rows")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_last_repeat_rows = inst.Scalar("int64", name="tail_last_repeat_rows")
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        input_outer_dim = self.tiling_param.input_shape[0]
        max_rows_in_ub.set_as(floor_align(self.ub_size // (input_inner_dim + 1), self.block_element))
        rows_each_core.set_as(self._ceil_32bytes_count(self._ceil_div(self.tiling_param.input_shape[0],
                                                                      self.aicore_num), self.block_element))
        aicore_num_used.set_as(self._ceil_div(input_outer_dim, rows_each_core))
        repeat_times.set_as(rows_each_core // max_rows_in_ub)
        last_repeat_rows.set_as(rows_each_core % max_rows_in_ub)
        tail_rows.set_as(input_outer_dim % rows_each_core)
        tail_rows_repeat_times.set_as(tail_rows // max_rows_in_ub)
        tail_last_repeat_rows.set_as(tail_rows % max_rows_in_ub)

        src_addr = inst.Scalar("int64", name="src_addr")
        dst_addr = inst.Scalar("int64", name="dst_addr")

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(self.dtype, (max_rows_in_ub * input_inner_dim,),
                                   scope=tik.scope_ubuf, name="input_ub")
            output_ub = inst.Tensor(self.dtype, (max_rows_in_ub,), scope=tik.scope_ubuf, name="output_ub")
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, repeat_times) as repeat_idx:
                            src_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                            input_inner_dim)
                            dst_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * 1)
                            self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, max_rows_in_ub,
                                                                      input_ub, output_ub)
                        with inst.if_scope(last_repeat_rows > 0):
                            src_addr.set_as((core_idx * rows_each_core + repeat_times * max_rows_in_ub) * \
                                            input_inner_dim)
                            dst_addr.set_as((core_idx * rows_each_core + repeat_times * max_rows_in_ub) * 1)
                            self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, last_repeat_rows,
                                                                      input_ub, output_ub)
                    with inst.else_scope():
                        with inst.for_range(0, tail_rows_repeat_times) as tail_repeat_idx:
                            src_addr.set_as((core_idx * rows_each_core + tail_repeat_idx * max_rows_in_ub) * \
                                            input_inner_dim)
                            dst_addr.set_as((core_idx * rows_each_core + tail_repeat_idx * max_rows_in_ub) * 1)
                            self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, max_rows_in_ub,
                                                                      input_ub, output_ub)
                        with inst.if_scope(tail_last_repeat_rows > 0):
                            src_addr.set_as((core_idx * rows_each_core + tail_rows_repeat_times * max_rows_in_ub) * \
                                            input_inner_dim)
                            dst_addr.set_as((core_idx * rows_each_core + tail_rows_repeat_times * max_rows_in_ub) * 1)
                            self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, tail_last_repeat_rows,
                                                                      input_ub, output_ub)
                with inst.else_scope():
                    with inst.for_range(0, repeat_times) as repeat_idx:
                        src_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * input_inner_dim)
                        dst_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * 1)
                        self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, max_rows_in_ub,
                                                                  input_ub, output_ub)
                    with inst.if_scope(last_repeat_rows > 0):
                        src_addr.set_as((core_idx * rows_each_core + repeat_times * max_rows_in_ub) * input_inner_dim)
                        dst_addr.set_as((core_idx * rows_each_core + repeat_times * max_rows_in_ub) * 1)
                        self._do_with_last_dim_equal_one_per_loop(src_addr, dst_addr, last_repeat_rows,
                                                                  input_ub, output_ub)

    def _do_with_last_dim_equal_one_per_loop(self, src_addr, dst_addr, cur_repeat_rows, input_ub, output_ub):
        """
        slice the last dim and the last dim size of outshape is equal one.
        Parameters
        ----------
        src_addr: the ub addr when move data from gm to ub
        dst_addr: the gm addr when move data from ub to gm
        cur_repeat_rows: number of rows processed at one loop
        input_ub: ub for storing input data
        output_ub: ub for storing output data

        Returns
        -------
        None
        """
        inst = self.tik_instance
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        start_num = inst.Scalar("int64", name="start_num")
        loop_data = inst.Scalar("int64", name="loop_data")
        start_num.set_as(self.tiling_param.begin[self.shape_length - 1])
        loop_data.set_as(cur_repeat_rows * input_inner_dim)
        self._data_move(input_ub[0], self.input_gm[src_addr], loop_data)
        with inst.for_range(0, cur_repeat_rows) as idx:
            output_ub[idx].set_as(input_ub[idx * input_inner_dim + start_num])
        self._data_move(self.output_gm[dst_addr], output_ub, cur_repeat_rows)

    def _do_with_one_dim(self, core_idx):
        """
        slice the data and the length of outshape is equal one.
        Parameters
        ----------
        core_idx: number of ai_core

        Returns
        -------
        None
        """
        inst = self.tik_instance
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        core_data = inst.Scalar("int64", name="core_data")
        tail_data = inst.Scalar("int64", name="tail_data")
        input_addr = inst.Scalar("int64", name="input_addr")
        output_addr = inst.Scalar("int64", name="output_addr")
        max_data_in_ub = inst.Scalar("int64", name="max_data_in_ub")
        output_inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        start_num = self.tiling_param.begin[self.shape_length - 1]
        core_data.set_as(ceil_align(ceil_div(output_inner_dim, self.aicore_num), self.block_element))
        aicore_num_used.set_as(ceil_div(output_inner_dim, core_data))
        max_data_in_ub.set_as(floor_align(self.ub_size, self.block_element))
        with inst.if_scope(aicore_num_used == 1):
            core_data.set_as(output_inner_dim)
        tail_data.set_as(output_inner_dim % core_data)
        with inst.if_scope(tail_data == 0):
            tail_data.set_as(core_data)
        with inst.new_stmt_scope():
            one_dim_ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="one_dim_ub")
            with inst.if_scope(core_idx < aicore_num_used - 1):
                input_addr.set_as(core_idx * core_data + start_num)
                output_addr.set_as(core_idx * core_data)
                self._do_with_one_dim_per_core(input_addr, output_addr, core_data, one_dim_ub, max_data_in_ub)
            with inst.elif_scope(core_idx == aicore_num_used - 1):
                input_addr.set_as((aicore_num_used - 1) * core_data + start_num)
                output_addr.set_as((aicore_num_used - 1) * core_data)
                self._do_with_one_dim_per_core(input_addr, output_addr, tail_data, one_dim_ub, max_data_in_ub)

    def _do_with_one_dim_per_core(self, input_addr, output_addr, core_data, input_ub, max_data_in_ub):
        """
        slice the data and the length of outshape is equal one.
        Parameters
        ----------
        input_addr: the ub addr when move data from gm to ub
        output_addr: the gm addr when move data from ub to gm
        core_data: number of data processed at one loop
        input_ub: ub for storing input data
        max_data_in_ub: maximum data stored in UB

        Returns
        -------
        None
        """
        inst = self.tik_instance
        src_addr = inst.Scalar("int64", name="src_addr")
        dst_addr = inst.Scalar("int64", name="dst_addr")
        loop_times = core_data // max_data_in_ub
        last_loop_data = core_data % max_data_in_ub
        with inst.for_range(0, loop_times) as loop_idx:
            src_addr.set_as(input_addr + loop_idx * max_data_in_ub)
            dst_addr.set_as(output_addr + loop_idx * max_data_in_ub)
            self._data_move(input_ub[0], self.input_gm[src_addr], max_data_in_ub)
            self._data_move(self.output_gm[dst_addr], input_ub[0], max_data_in_ub)
        src_addr.set_as(input_addr + loop_times * max_data_in_ub)
        dst_addr.set_as(output_addr + loop_times * max_data_in_ub)
        with inst.if_scope(last_loop_data > 0):
            self._data_move(input_ub[0], self.input_gm[src_addr], last_loop_data)
            self._data_move(self.output_gm[dst_addr], input_ub[0], last_loop_data)

    def _do_only_slice_last_dim_with_datamove(self, core_idx):
        """
        slice the last dim with data_move instruction.
        Parameters
        ----------
        core_idx: number of ai_core

        Returns
        -------
        None
        """
        inst = self.tik_instance
        max_rows_in_ub = inst.Scalar("int64", name="max_rows_in_ub")
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        last_repeat_rows = inst.Scalar("int64", name="last_repeat_rows")
        tail_repeat_rows = inst.Scalar("int64", name="tail_repeat_rows")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_rows_last_repeat = inst.Scalar("int64", name="tail_rows_last_repeat")
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        input_addr = inst.Scalar("int64", name="input_addr")
        output_addr = inst.Scalar("int64", name="output_addr")
        max_rows_in_ub.set_as(floor_align(self.ub_size // self.tiling_param.input_shape[self.shape_length - 1],
                                          self.block_element))
        with inst.if_scope(max_rows_in_ub // self.block_element > MAX_REPEAT):
            max_rows_in_ub.set_as(MAX_REPEAT * self.block_element)
        rows_each_core.set_as(self._ceil_32bytes_count(self._ceil_div(self.tiling_param.input_shape[0],
                                                                      self.aicore_num), self.block_element))
        repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
        last_repeat_rows.set_as(rows_each_core % max_rows_in_ub)
        with inst.if_scope(last_repeat_rows == 0):
            last_repeat_rows.set_as(max_rows_in_ub)
        tail_rows.set_as(self.tiling_param.input_shape[0] % rows_each_core)
        tail_rows_repeat_times.set_as(self._ceil_div(tail_rows, max_rows_in_ub))
        tail_repeat_rows.set_as(floor_align(tail_rows % max_rows_in_ub, self.block_element))
        tail_rows_last_repeat.set_as(tail_rows % max_rows_in_ub - tail_repeat_rows)
        with inst.if_scope(tik.all(tail_rows_last_repeat == 0, tail_repeat_rows == 0)):
            tail_repeat_rows.set_as(max_rows_in_ub)
        aicore_num_used.set_as(self._ceil_div(self.tiling_param.input_shape[0], rows_each_core))
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        output_inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        with inst.new_stmt_scope():
            input_ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="input_ub")
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, repeat_times - 1) as repeat_idx:
                            input_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                              input_inner_dim)
                            output_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                               output_inner_dim)
                            self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, max_rows_in_ub)
                        input_addr.set_as((core_idx * rows_each_core + (repeat_times - 1) * max_rows_in_ub) * \
                                              input_inner_dim)
                        output_addr.set_as((core_idx * rows_each_core + (repeat_times - 1) * max_rows_in_ub) * \
                                            output_inner_dim)
                        self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, last_repeat_rows)
                    with inst.else_scope():
                        with inst.for_range(0, tail_rows_repeat_times - 1) as tail_repeat_idx:
                            input_addr.set_as(((aicore_num_used - 1) * rows_each_core + \
                                               tail_repeat_idx * max_rows_in_ub) * input_inner_dim)
                            output_addr.set_as(((aicore_num_used - 1) * rows_each_core + \
                                                tail_repeat_idx * max_rows_in_ub) * output_inner_dim)
                            self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, max_rows_in_ub)
                        with inst.if_scope(tail_repeat_rows > 0):
                            input_addr.set_as(((aicore_num_used - 1) * rows_each_core + \
                                               (tail_rows_repeat_times - 1) * max_rows_in_ub) * input_inner_dim)
                            output_addr.set_as(((aicore_num_used - 1) * rows_each_core + \
                                                (tail_rows_repeat_times - 1) * max_rows_in_ub) * output_inner_dim)
                            self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, tail_repeat_rows)
                        with inst.if_scope(tail_rows_last_repeat > 0):
                            self._do_with_data_move_tail_rows((aicore_num_used - 1) * rows_each_core + \
                                                            (tail_rows_repeat_times -1) * max_rows_in_ub + \
                                                            tail_repeat_rows, tail_rows_last_repeat, input_ub)
                with inst.else_scope():
                    with inst.if_scope(core_idx < aicore_num_used):
                        with inst.for_range(0, repeat_times - 1) as repeat_idx:
                            input_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                              input_inner_dim)
                            output_addr.set_as((core_idx * rows_each_core + repeat_idx * max_rows_in_ub) * \
                                               output_inner_dim)
                            self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, max_rows_in_ub)
                        input_addr.set_as((core_idx * rows_each_core + (repeat_times - 1) * max_rows_in_ub) * \
                                              input_inner_dim)
                        output_addr.set_as((core_idx * rows_each_core + (repeat_times - 1) * max_rows_in_ub) * \
                                            output_inner_dim)
                        self._do_with_data_move_per_loop(input_addr, output_addr, input_ub, last_repeat_rows)

    def _do_with_data_move_per_loop(self, input_addr, output_addr, input_ub, cur_repeat_rows):
        """
        slice the last dim with data_move instruction.
        Parameters
        ----------
        input_addr: the ub addr when move data from gm to ub
        output_addr: the gm addr when move data from ub to gm
        input_ub: ub for storing input data
        cur_repeat_rows: number of rows processed at one loop

        Returns
        -------
        None
        """
        inst = self.tik_instance
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        output_inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        output_last_dim_part1 = output_inner_dim // self.block_element * self.block_element
        output_last_dim_part1_block = output_last_dim_part1 // self.block_element
        output_last_dim_part2 = output_inner_dim - output_last_dim_part1
        start_num = self.tiling_param.begin[self.shape_length - 1]
        src_addr = inst.Scalar("int64", name="src_addr")
        dst_addr = inst.Scalar("int64", name="dst_addr")
        with inst.for_range(0, self.block_element) as loop_idx:
            src_addr.set_as(input_addr + loop_idx * input_inner_dim + start_num)
            dst_addr.set_as(loop_idx * output_last_dim_part1)
            inst.data_move(input_ub[dst_addr],
                           self.input_gm[src_addr],
                           0,
                           cur_repeat_rows // self.block_element,
                           output_last_dim_part1_block,
                           input_inner_dim - output_last_dim_part1_block,
                           output_last_dim_part1 - output_last_dim_part1_block)
            src_addr.set_as(loop_idx * output_last_dim_part1)
            dst_addr.set_as(output_addr + loop_idx * output_inner_dim)
            inst.data_move(self.output_gm[dst_addr],
                           input_ub[src_addr],
                           0,
                           cur_repeat_rows // self.block_element,
                           output_last_dim_part1_block,
                           output_last_dim_part1 - output_last_dim_part1_block,
                           output_inner_dim - output_last_dim_part1_block)
            with inst.if_scope(output_last_dim_part2 > 0):
                inst.data_move(input_ub[loop_idx * self.block_element],
                               self.input_gm[input_addr + loop_idx * input_inner_dim +
                                             start_num +
                                             (output_inner_dim - self.block_element)],
                               0,
                               cur_repeat_rows // self.block_element,
                               1,
                               input_inner_dim - 1,
                               self.block_element - 1)
                inst.data_move(self.output_gm[output_addr + loop_idx * output_inner_dim +
                                              (output_inner_dim - self.block_element)],
                               input_ub[loop_idx * self.block_element],
                               0,
                               cur_repeat_rows // self.block_element,
                               1,
                               self.block_element -1,
                               output_inner_dim - 1)

    def _do_with_data_move_tail_rows(self, rows, tail_rows_last_repeat, input_ub):
        """
        slice the tail data with data_move instruction.
        Parameters
        ----------
        rows: start data row
        tail_rows_last_repeat: rows to be processed
        input_ub: ub for storing input data

        Returns
        -------
        None
        """
        inst = self.tik_instance
        input_inner_dim = self.tiling_param.input_shape[self.shape_length - 1]
        output_inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        output_inner_dim_block = self._ceil_div(output_inner_dim, self.block_element)
        start_num = self.tiling_param.begin[self.shape_length - 1]
        input_addr = rows * input_inner_dim
        output_addr = rows * output_inner_dim
        with inst.for_range(0, tail_rows_last_repeat) as loop_idx:
            inst.data_move(input_ub,
                           self.input_gm[input_addr + loop_idx * input_inner_dim + start_num],
                           0, 1, output_inner_dim_block, 0, 0)
            inst.data_move(self.output_gm[output_addr + loop_idx * output_inner_dim],
                           input_ub, 0, 1, output_inner_dim_block, 0, 0)

    def _do_only_slice_last_dim_with_vnchwconv(self, core_idx):
        """
        slice the last dim and a column ub can fit multiple rows of input data.
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        dtype_size = self.dtype_size
        output_shape = self.tiling_param.output_shape
        input_shape = self.tiling_param.input_shape
        shape_length = self.tiling_param.shape_length
        float16_dtype_size = common_util.get_data_size("float16")
        multi_times = dtype_size // float16_dtype_size
        tensor_dtype = "float16"
        vnchwconv_column = 16
        input_inner_dim = input_shape[shape_length - 1] * multi_times
        output_inner_dim = output_shape[shape_length - 1] * multi_times
        out_dim = self.tiling_param.out_dim
        ub_size = self.ub_size * multi_times // 2
        begin_value = self.tiling_param.begin[shape_length - 1] * multi_times
        element_each_block = self.block_element * multi_times
        output_32bytes_align_rows = inst.Scalar("int64", name="output_32bytes_align_rows",
                                                init_value=element_each_block)
        with inst.if_scope(output_32bytes_align_rows % output_inner_dim == 0):
            output_32bytes_align_rows.set_as(output_32bytes_align_rows // output_inner_dim)
        with inst.elif_scope(output_inner_dim % output_32bytes_align_rows == 0):
            output_32bytes_align_rows.set_as(1)
        max_rows_in_ub = floor_align(ub_size // (input_inner_dim * vnchwconv_column), output_32bytes_align_rows)
        rows_each_core = inst.Scalar("int64", name="rows_each_core")
        repeat_times = inst.Scalar("int64", name="repeat_times")
        rows_each_repeat = inst.Scalar("int64", name="rows_each_repeat")
        aicore_num_used = inst.Scalar("int64", name="aicore_num_used")
        tail_rows = inst.Scalar("int64", name="tail_rows")
        tail_rows_repeat_times = inst.Scalar("int64", name="tail_rows_repeat_times")
        tail_rows_repeat_tail_count = inst.Scalar("int64", name="tail_rows_repeat_tail_count")
        tail_rows_repeat_roll_back_rows = inst.Scalar("int64", name="tail_rows_repeat_roll_back_rows", init_value=0)
        rows_each_core.set_as(self._ceil_32bytes_count(self._ceil_div(out_dim, core_num), output_32bytes_align_rows))
        repeat_times.set_as(self._ceil_div(rows_each_core, max_rows_in_ub))
        rows_each_repeat.set_as(self._ceil_32bytes_count(self._ceil_div(rows_each_core, repeat_times),
                                                         output_32bytes_align_rows))
        rows_each_core.set_as(rows_each_repeat * repeat_times)
        aicore_num_used.set_as(self._ceil_div(out_dim, rows_each_core))
        with inst.if_scope(aicore_num_used == 1):
            rows_each_core.set_as(out_dim)
            rows_each_repeat.set_as(out_dim)
        loop_times = self._ceil_div(repeat_times, 16)
        last_loop_rows = inst.Scalar("int64", name="last_loop_rows", init_value=repeat_times % 16)
        with inst.if_scope(last_loop_rows == 0):
            last_loop_rows.set_as(16)

        tail_rows.set_as(out_dim % rows_each_core)
        tail_rows_repeat_times.set_as(self._ceil_div(tail_rows, rows_each_repeat))
        tail_rows_repeat_tail_count.set_as(tail_rows % rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count == 0):
            tail_rows_repeat_tail_count.set_as(rows_each_repeat)
        with inst.if_scope(tail_rows_repeat_tail_count % output_32bytes_align_rows != 0):
            tail_rows_repeat_roll_back_rows.set_as(self._ceil_32bytes_count(tail_rows_repeat_tail_count,
                                                                            output_32bytes_align_rows) - \
                                                   tail_rows_repeat_tail_count)
        tail_loop_times = self._ceil_div(tail_rows_repeat_times - 1, 16)
        tail_last_loop_rows = inst.Scalar("int64", name="tail_last_loop_rows",
                                          init_value=(tail_rows_repeat_times -1) % 16)
        with inst.if_scope(tail_last_loop_rows == 0):
            tail_last_loop_rows.set_as(16)

        if dtype_size % float16_dtype_size == 0:
            input_gm = self.input_gm.reinterpret_cast_to("float16")
            output_gm = self.output_gm.reinterpret_cast_to("float16")
        else:
            input_gm = self.input_gm
            output_gm = self.output_gm

        input_addr = inst.Scalar("int64", name="input_addr")
        output_addr = inst.Scalar("int64", name="output_addr")
        param_dict = {"input_gm": input_gm, "output_gm": output_gm, "rows_each_repeat": rows_each_repeat,
                      "input_inner_dim": input_inner_dim, "output_inner_dim": output_inner_dim,
                      "begin_value": begin_value, "element_each_block": element_each_block}

        with inst.new_stmt_scope():
            input_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="input_ub")
            vnchw_conv_ub = inst.Tensor(tensor_dtype, (ub_size,), scope=tik.scope_ubuf, name="vnchw_conv_ub")
            param_dict["input_ub"] = input_ub
            param_dict["vnchw_conv_ub"] = vnchw_conv_ub
            with inst.if_scope(core_idx < aicore_num_used):
                with inst.if_scope(tail_rows != 0):
                    with inst.if_scope(core_idx < aicore_num_used - 1):
                        with inst.for_range(0, loop_times - 1) as loop_idx:
                            input_addr.set_as(rows_each_core * input_inner_dim * core_idx + \
                                              loop_idx * rows_each_repeat * input_inner_dim * 16)
                            output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                               loop_idx * rows_each_repeat * output_inner_dim * 16)
                            param_dict["input_addr"] = input_addr
                            param_dict["output_addr"] = output_addr
                            param_dict["loop_rows"] = 16
                            self._do_with_vnchwconv_per_loop(param_dict)
                        input_addr.set_as(rows_each_core * input_inner_dim * core_idx + \
                                          (loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                        output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                           (loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = last_loop_rows
                        self._do_with_vnchwconv_per_loop(param_dict)
                    with inst.else_scope():
                        with inst.if_scope(tail_rows_repeat_times > 1):
                            with inst.for_range(0, tail_loop_times - 1) as loop_idx:
                                input_addr.set_as(rows_each_core * input_inner_dim * (aicore_num_used - 1) + \
                                                  loop_idx * rows_each_repeat * input_inner_dim * 16)
                                output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) + \
                                                   loop_idx * rows_each_repeat * output_inner_dim * 16)
                                param_dict["input_addr"] = input_addr
                                param_dict["output_addr"] = output_addr
                                param_dict["loop_rows"] = 16
                                self._do_with_vnchwconv_per_loop(param_dict)
                            input_addr.set_as(rows_each_core * input_inner_dim * (aicore_num_used - 1) + \
                                              (tail_loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                            output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) + \
                                               (tail_loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                            param_dict["input_addr"] = input_addr
                            param_dict["output_addr"] = output_addr
                            param_dict["loop_rows"] = tail_last_loop_rows
                            self._do_with_vnchwconv_per_loop(param_dict)
                        input_addr.set_as(rows_each_core * input_inner_dim * (aicore_num_used - 1) + \
                                          (tail_rows_repeat_times -1) * rows_each_repeat * input_inner_dim - \
                                          tail_rows_repeat_roll_back_rows * input_inner_dim)
                        output_addr.set_as(rows_each_core * output_inner_dim * (aicore_num_used - 1) + \
                                           (tail_rows_repeat_times -1) * rows_each_repeat * output_inner_dim - \
                                           tail_rows_repeat_roll_back_rows * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = 1
                        param_dict["rows_each_repeat"] = tail_rows_repeat_tail_count + tail_rows_repeat_roll_back_rows
                        self._do_with_vnchwconv_per_loop(param_dict)
                with inst.else_scope():
                    with inst.for_range(0, loop_times - 1) as loop_idx:
                        input_addr.set_as(rows_each_core * input_inner_dim * core_idx + \
                                          loop_idx * rows_each_repeat * input_inner_dim * 16)
                        output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                           loop_idx * 16 * rows_each_repeat * output_inner_dim)
                        param_dict["input_addr"] = input_addr
                        param_dict["output_addr"] = output_addr
                        param_dict["loop_rows"] = 16
                        self._do_with_vnchwconv_per_loop(param_dict)
                    input_addr.set_as(rows_each_core * input_inner_dim * core_idx + \
                                      (loop_times - 1) * rows_each_repeat * input_inner_dim * 16)
                    output_addr.set_as(rows_each_core * output_inner_dim * core_idx + \
                                       (loop_times - 1) * 16 * rows_each_repeat * output_inner_dim)
                    param_dict["input_addr"] = input_addr
                    param_dict["output_addr"] = output_addr
                    param_dict["loop_rows"] = last_loop_rows
                    self._do_with_vnchwconv_per_loop(param_dict)

    def _do_with_vnchwconv_per_loop(self, param_dict):
        """
        slice multiple rows of input data at one loop
        """
        inst = self.tik_instance
        input_gm = param_dict["input_gm"]
        output_gm = param_dict["output_gm"]
        rows_each_repeat = param_dict["rows_each_repeat"]
        input_inner_dim = param_dict["input_inner_dim"]
        output_inner_dim = param_dict["output_inner_dim"]
        begin_value = param_dict["begin_value"]
        element_each_block = param_dict["element_each_block"]
        input_addr = param_dict["input_addr"]
        output_addr = param_dict["output_addr"]
        loop_rows = param_dict["loop_rows"]
        input_ub = param_dict["input_ub"]
        vnchw_conv_ub = param_dict["vnchw_conv_ub"]
        vnchw_conv_repeat_times = inst.Scalar("int64", name="vnchw_conv_repeat_times")
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block))
        loop_count = inst.Scalar("int64", name="loop_count")
        loop_count.set_as((self._ceil_div(rows_each_repeat * input_inner_dim, element_each_block)) * \
                          element_each_block)
        with inst.for_range(0, loop_rows) as loop_rows_idx:
            src_addr_in = input_addr + loop_rows_idx * rows_each_repeat * input_inner_dim
            dst_addr_in = loop_rows_idx * ceil_align(rows_each_repeat * input_inner_dim, element_each_block)
            self._data_move(input_ub[dst_addr_in], input_gm[src_addr_in], rows_each_repeat * input_inner_dim)
        self._do_with_input2vnchwconv(vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times,
                                      element_each_block)
        inst.data_move(input_ub, vnchw_conv_ub[begin_value * 16], 0, rows_each_repeat,
                       output_inner_dim, input_inner_dim - output_inner_dim, 0)
        vnchw_conv_repeat_times.set_as(self._ceil_div(rows_each_repeat * output_inner_dim, element_each_block))
        self._do_with_vnchwconv2output(vnchw_conv_ub, input_ub, vnchw_conv_repeat_times, element_each_block)
        self._data_move(output_gm[output_addr], vnchw_conv_ub, loop_rows * rows_each_repeat * output_inner_dim)

    def _do_with_vnchwconv2output(self, vnchw_conv_ub, input_ub, vnchw_conv_repeat_times, element_each_block):
        """
        rearrange data in ub with vnchwconv instruction
        """
        inst = self.tik_instance
        dst_list = [vnchw_conv_ub[i * vnchw_conv_repeat_times * element_each_block] for i in range(16)]
        src_list = [input_ub[i * element_each_block] for i in range(16)]
        with inst.if_scope(vnchw_conv_repeat_times == 1):
            inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
        with inst.else_scope():
            inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 1, 16)

    def _do_with_input2vnchwconv(self, vnchw_conv_ub, input_ub, loop_count, vnchw_conv_repeat_times,
                                 element_each_block):
        """
        rearrange data in ub with vnchwconv instruction
        """
        inst = self.tik_instance
        dst_list = [vnchw_conv_ub[i * element_each_block] for i in range(16)]
        src_list = [input_ub[i * loop_count] for i in range(16)]
        with inst.if_scope(vnchw_conv_repeat_times == 1):
            inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
        with inst.else_scope():
            inst.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 16, 1)

    def _do_large_last_dim_multi_rows(self, core_idx):
        """
        _do_large_last_dim_multi_rows
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        # out_dim = 67, num_rows_per_aicore: 3, 3, 3, 2, 2, 2 ...
        num_rows_per_aicore = inst.Scalar("int64", name="num_rows_per_aicore")
        num_tail_rows = inst.Scalar("int64", name="num_tail_rows")
        num_rows_per_aicore.set_as(self.tiling_param.out_dim // core_num)
        num_tail_rows.set_as(self.tiling_param.out_dim - core_num * num_rows_per_aicore)

        row_idx = inst.Scalar("int64", name="row_idx")
        with inst.if_scope(core_idx < num_tail_rows):
            row_idx.set_as(core_idx + core_idx * num_rows_per_aicore)
        with inst.else_scope():
            row_idx.set_as(num_tail_rows + core_idx * num_rows_per_aicore)

        with inst.if_scope(core_idx < num_tail_rows):
            num_rows_per_aicore.set_as(num_rows_per_aicore + 1)

        with inst.if_scope(row_idx < self.tiling_param.out_dim):
            self._do_large_last_dim_multi_rows_per_aicore(row_idx, num_rows_per_aicore)

    def _do_large_last_dim_multi_rows_per_aicore(self, row_idx, num_rows):
        """
        _do_large_last_dim_multi_rows_per_aicore
        """
        inst = self.tik_instance
        output_shape = self.tiling_param.output_shape
        input_shape = self.tiling_param.input_shape
        dim_count = self.tiling_param.shape_length
        max_rows_per_data_move = inst.Scalar("int64", name="loops")
        max_rows_per_data_move.set_as(self.ub_size // output_shape[dim_count - 1])
        loops = inst.Scalar("int64", name="loops")
        loops.set_as(num_rows // max_rows_per_data_move)

        row_loop_idx = inst.Scalar("int64", name="row_loop_idx")

        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="ub")
            with inst.for_range(0, loops, name="loops_for_range") as loop_idx:
                row_loop_idx.set_as(row_idx + loop_idx * max_rows_per_data_move)
                input_gm_addr = self._get_input_gm_addr(row_loop_idx)
                output_gm_addr = self._get_output_gm_addr(row_loop_idx)
                inst.data_move(
                    ub,
                    self.input_gm[input_gm_addr],
                    0,
                    max_rows_per_data_move,
                    output_shape[dim_count - 1] // self.block_element,
                    (input_shape[dim_count - 1] - output_shape[dim_count - 1]) // self.block_element,
                    0)  # input gm -> ub

                inst.data_move(
                    self.output_gm[output_gm_addr],
                    ub,
                    0,
                    1,
                    max_rows_per_data_move * output_shape[dim_count - 1] // self.block_element,
                    0,
                    0)  # ub -> output gm

            num_remain_rows = num_rows - loops * max_rows_per_data_move
            with inst.if_scope(num_remain_rows > 0):
                row_loop_idx.set_as(row_idx + loops * max_rows_per_data_move)
                input_gm_addr = self._get_input_gm_addr(row_loop_idx)
                output_gm_addr = self._get_output_gm_addr(row_loop_idx)
                inst.data_move(
                    ub,
                    self.input_gm[input_gm_addr],
                    0,
                    num_remain_rows,
                    output_shape[dim_count - 1] // self.block_element,
                    (input_shape[dim_count - 1] - output_shape[dim_count - 1]) // self.block_element,
                    0)  # input gm -> ub
                inst.data_move(
                    self.output_gm[output_gm_addr],
                    ub,
                    0,
                    1,
                    num_remain_rows * output_shape[dim_count - 1] // self.block_element,
                    0,
                    0)  # ub -> output gm

    def _do_large_last_dim(self, core_idx):
        self._do_large_last_dim_normal(core_idx)

    def _do_large_last_dim_normal(self, core_idx):
        """
        _do_large_last_dim_normal
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        output_shape = self.tiling_param.output_shape
        inner_loops = self._ceil_div(output_shape[self.shape_length - 1], self.ub_size)
        out_loops = self._ceil_div(self.tiling_param.out_dim, core_num)
        with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
            idx = core_idx * out_loops + loop_idx
            with inst.if_scope(idx < self.tiling_param.out_dim):
                input_gm_addr = self._get_input_gm_addr(idx)
                output_gm_addr = self._get_output_gm_addr(idx)
                with inst.for_range(0, inner_loops, name="inner_loop") as inner_loop_idx:
                    with inst.if_scope(output_shape[self.shape_length - 1] % self.block_element == 0):
                        self._do_large_last_dim_align(input_gm_addr, output_gm_addr, inner_loop_idx)
                    with inst.else_scope():
                        self._do_large_last_dim_not_align(input_gm_addr, output_gm_addr, inner_loop_idx)

    # pylint: disable=invalid-name
    def _do_small_last_dim(self, core_idx):
        """
        _do_small_last_dim
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        output_shape = self.tiling_param.output_shape
        inner_dim = output_shape[self.shape_length - 1]
        out_dim = self.tiling_param.out_dim
        out_loops = self._ceil_div(out_dim, core_num)
        tmp_ub_size = self.block_element
        ub_size = self.ub_size - self.block_element
        ub_data_count = inst.Scalar("int32", name="out_ub_data_count")
        ub_data_count.set_as(0)
        input_gm = self.input_gm
        output_gm = self.output_gm
        need_update_out_addr = inst.Scalar("int32", name="need_update_out_addr")
        need_update_out_addr.set_as(1)
        output_gm_addr = inst.Scalar(self.tiling_param.dtype,
                                     name="output_addr")
        with inst.new_stmt_scope():
            tmp_ub = inst.Tensor(self.dtype, (tmp_ub_size,), scope=tik.scope_ubuf, name="tmp_ub")
            ub = inst.Tensor(self.dtype, (ub_size,), scope=tik.scope_ubuf, name="out_ub")
            with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
                idx = core_idx * out_loops + loop_idx
                with inst.if_scope(idx < self.tiling_param.out_dim):
                    input_gm_addr = self._get_input_gm_addr(idx)
                    with inst.if_scope(need_update_out_addr == 1):
                        need_update_out_addr.set_as(0)
                        output_gm_addr.set_as(self._get_output_gm_addr(idx))

                    with inst.if_scope(ub_data_count + inner_dim > ub_size):
                        self._data_move(output_gm[output_gm_addr], ub, ub_data_count)
                        ub_data_count.set_as(0)
                        output_gm_addr.set_as(self._get_output_gm_addr(idx))
                    self._data_move(tmp_ub, input_gm[input_gm_addr], self.block_element)

                    with inst.for_range(0, inner_dim) as index:
                        ub[ub_data_count + index] = tmp_ub[index]
                    ub_data_count.set_as(ub_data_count + inner_dim)

                    with inst.if_scope(loop_idx == out_loops - 1):
                        self._add_tail(ub, tmp_ub, idx, ub_data_count)
            with inst.if_scope(ub_data_count != 0):
                self._data_move(output_gm[output_gm_addr], ub, ub_data_count)

    def _do_large_last_dim_align(self, input_gm_addr, output_gm_addr, inner_loop_idx):
        """
        _do_large_last_dim_align
        """
        inst = self.tik_instance
        total = self.tiling_param.output_shape[self.shape_length - 1]
        input_gm = self.input_gm
        output_gm = self.output_gm
        count = inst.Scalar("int32", name="remain")
        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="out_ub")
            count.set_as(total - self.ub_size * inner_loop_idx)
            with inst.if_scope(count > self.ub_size):
                count.set_as(self.ub_size)

            self._data_move(ub, input_gm[input_gm_addr + inner_loop_idx * self.ub_size], count)
            self._data_move(output_gm[output_gm_addr + inner_loop_idx * self.ub_size], ub, count)

    # pylint: disable=too-many-locals,invalid-name
    def _do_large_last_dim_not_align(self, input_gm_addr, output_gm_addr, inner_loop_idx):
        """
        _do_large_last_dim_not_align
        """
        inst = self.tik_instance
        total = self.tiling_param.output_shape[self.shape_length - 1]
        input_gm = self.input_gm
        output_gm = self.output_gm
        count = inst.Scalar("int32", name="remain")
        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="out_ub")
            count.set_as(total - self.ub_size * inner_loop_idx)
            with inst.if_scope(count >= self.ub_size):
                self._data_move(ub, input_gm[input_gm_addr + inner_loop_idx * self.ub_size], self.ub_size)
                self._data_move(output_gm[output_gm_addr + inner_loop_idx * self.ub_size], ub, self.ub_size)
            with inst.else_scope():
                with inst.if_scope(inner_loop_idx > 0):
                    align_count = self._ceil_32bytes_count(count, self.block_element)
                    redundant_count = align_count - count
                    new_in_start_index = (input_gm_addr + inner_loop_idx * self.ub_size - redundant_count)
                    new_out_start_index = (output_gm_addr + inner_loop_idx * self.ub_size - redundant_count)
                    self._data_move(ub, input_gm[new_in_start_index:], align_count)
                    self._data_move(output_gm[new_out_start_index:], ub, align_count)
                with inst.else_scope():
                    in_start_index = (input_gm_addr + inner_loop_idx * self.ub_size)
                    out_start_index = (output_gm_addr + inner_loop_idx * self.ub_size)
                    self._data_move(ub, input_gm[in_start_index:], self.block_element)
                    self._data_move(output_gm[out_start_index:], ub, self.block_element)

                    in_start_index += self.block_element
                    out_start_index += self.block_element
                    align_count = self._ceil_32bytes_count(count - self.block_element, self.block_element)
                    redundant_count = align_count - count + self.block_element
                    new_in_start_index = in_start_index - redundant_count
                    new_out_start_index = out_start_index - redundant_count
                    self._data_move(ub, input_gm[new_in_start_index:], align_count)
                    self._data_move(output_gm[new_out_start_index:], ub, align_count)

    def _add_tail(self, ub, tmp_ub, idx, ub_data_count):
        """
        _add_tail
        """
        inst = self.tik_instance
        inner_dim = self.tiling_param.output_shape[self.shape_length - 1]
        out_dim = self.tiling_param.out_dim
        align_count = self._ceil_32bytes_count(ub_data_count, self.block_element)
        overlap_count = align_count - ub_data_count
        ext_rows = self._ceil_div(overlap_count, inner_dim)
        input_gm = self.input_gm
        with inst.for_range(1, ext_rows + 1, name="ext_row") as row_idx:
            with inst.if_scope(idx + row_idx < out_dim):
                input_addr = self._get_input_gm_addr(idx + row_idx)
                self._data_move(tmp_ub, input_gm[input_addr], self.block_element)
                with inst.for_range(0, inner_dim) as index:
                    with inst.if_scope(ub_data_count < align_count):
                        ub[ub_data_count] = tmp_ub[index]
                        ub_data_count.set_as(ub_data_count + 1)

    def _do_small_last_dim_with_vnchwconv(self, core_idx):
        """
        _do_small_last_dim_with_vnchwconv
        """
        inst = self.tik_instance
        core_num = self.aicore_num
        input_shape = self.tiling_param.input_shape
        output_shape = self.tiling_param.output_shape
        out_loops = self._ceil_div(self.tiling_param.out_dim_with_vnchwconv, core_num)
        compute_rows_each_inner_loops = self.ub_size_with_vnchwconv // (
            16 * input_shape[self.shape_length - 1]) // 16 * 16

        inner_loops = self._ceil_div(output_shape[self.shape_length - 2], compute_rows_each_inner_loops) - 1
        compute_rows_tail = output_shape[self.shape_length - 2] - inner_loops * compute_rows_each_inner_loops
        with inst.new_stmt_scope():
            ub1 = inst.Tensor(self.dtype, (self.ub_size_with_vnchwconv,), scope=tik.scope_ubuf, name="ub1")
            ub2 = inst.Tensor(self.dtype, (self.ub_size_with_vnchwconv,), scope=tik.scope_ubuf, name="ub2")
            with inst.for_range(0, out_loops, name="out_loops") as out_loops_idx:
                idx = core_idx * out_loops + out_loops_idx
                with inst.if_scope(idx < self.tiling_param.out_dim_with_vnchwconv):
                    input_gm_base_addr = self._get_input_base_gm_addr_with_vnchwconv(idx)
                    output_gm_base_addr = self._get_output_base_gm_addr_with_vnchwconv(idx)
                    with inst.for_range(0, inner_loops, name="inner_loops") as inner_loops_idx:
                        input_gm_addr = input_gm_base_addr + inner_loops_idx * \
                            compute_rows_each_inner_loops * input_shape[self.shape_length - 1]
                        output_gm_addr = output_gm_base_addr + inner_loops_idx * \
                            compute_rows_each_inner_loops * output_shape[self.shape_length - 1]

                        self._do_each_matrix_align(input_gm_addr, output_gm_addr, compute_rows_each_inner_loops, \
                                                   ub1, ub2)

                    input_gm_addr = input_gm_base_addr + inner_loops * compute_rows_each_inner_loops * input_shape[
                        self.shape_length - 1]
                    output_gm_addr = output_gm_base_addr + inner_loops * compute_rows_each_inner_loops * output_shape[
                        self.shape_length - 1]
                    param_list = [input_gm_addr, output_gm_addr, compute_rows_tail, out_loops_idx, out_loops, ub1, ub2]
                    self._do_each_matrix_tail(param_list)

    def _get_input_base_gm_addr_with_vnchwconv(self, cur_index: tik.Scalar):
        """
        _get_input_base_gm_addr_vnchw
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="input_addr")
        addr.set_as(self.tiling_param.begin[dim_count - 2] * self.tiling_param.input_shape[dim_count - 1])
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(3, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.input_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * (tmp + self.tiling_param.begin[dim_count - dim_idx]))
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _get_output_base_gm_addr_with_vnchwconv(self, cur_index: tik.Scalar):
        """
        _get_output_base_gm_addr_vnchw
        """
        inst = self.tik_instance
        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        dim_count = self.tiling_param.shape_length
        addr = inst.Scalar(self.tiling_param.dtype, name="output_addr")
        addr.set_as(0)
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")

        with inst.for_range(3, dim_count + 1) as dim_idx:
            dim = self.tiling_param.output_shape[dim_count - dim_idx]
            step = self.tiling_param.output_steps[dim_idx - 2]
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * tmp)
            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _do_each_matrix_align(self, input_gm_addr, output_gm_addr, rows, ub1, ub2):
        """
        _do_each_matrix_align
        """
        inst = self.tik_instance
        output_matrix_count = self.tiling_param.output_shape[self.shape_length - 1] * rows
        self._do_each_matrix_except_move_output_gm(input_gm_addr, output_gm_addr, rows, [ub1, ub2])
        self._data_move(self.output_gm[output_gm_addr], ub2, output_matrix_count)

    def _do_each_matrix_except_move_output_gm(self, input_gm_addr, output_gm_addr, rows, ub_list):
        """
        _do_each_matrix_except_move_output_gm
        """
        inst = self.tik_instance
        input_shape = self.tiling_param.input_shape
        output_shape = self.tiling_param.output_shape
        begin = self.tiling_param.begin
        ub1, ub2 = ub_list
        input_matrix_count = input_shape[self.shape_length - 1] * rows
        self._data_move(ub1, self.input_gm[input_gm_addr], input_matrix_count)

        # first vnchwconv_loop: ub1(32, 31) -> ub2(32 * 31, 16)
        vnchwconv_loop = self._ceil_div(input_matrix_count, 16)
        with inst.for_range(0, vnchwconv_loop) as i:
            src_addr = [ub1[16 * i + 16 * j] for j in range(16)]
            dst_addr = [ub2[16 * 16 * i + 16 * j] for j in range(16)]
            inst.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

        nburst_loop = rows // MAX_NBURST
        with inst.for_range(0, nburst_loop) as i:
            inst.data_move(
                ub1[i * MAX_NBURST * output_shape[self.shape_length - 1] * 16],
                ub2[(i * MAX_NBURST * input_shape[self.shape_length - 1] + begin[self.shape_length - 1]) * 16],
                0,
                MAX_NBURST,
                output_shape[self.shape_length - 1],
                input_shape[self.shape_length - 1] - output_shape[self.shape_length - 1],
                0)

        with inst.if_scope(rows % MAX_NBURST != 0):
            inst.data_move(
                ub1[nburst_loop * MAX_NBURST * output_shape[self.shape_length - 1] * 16],
                ub2[(nburst_loop * MAX_NBURST * input_shape[self.shape_length - 1] + begin[
                    self.shape_length - 1]) * 16],
                0,
                rows % MAX_NBURST,
                output_shape[self.shape_length - 1],
                input_shape[self.shape_length - 1] - output_shape[self.shape_length - 1],
                0)

        output_matrix_count = output_shape[self.shape_length - 1] * rows
        vnchwconv_loop = self._ceil_div(output_matrix_count, 16)
        with inst.for_range(0, vnchwconv_loop) as i:
            src_addr = [ub1[16 * 16 * i + 16 * j] for j in range(16)]
            dst_addr = [ub2[16 * i + 16 * j] for j in range(16)]
            inst.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

    def _do_each_matrix_tail(self, param_list):
        """
        _do_each_matrix_tail
        """
        inst = self.tik_instance
        input_gm_addr, output_gm_addr, rows, out_loops_idx, out_loops, ub1, ub2 = param_list
        output_matrix_count = self.tiling_param.output_shape[self.shape_length - 1] * rows
        ub_block = inst.Tensor(self.dtype, (self.block_element,), scope=tik.scope_ubuf, name="ub_block")

        self._do_each_matrix_except_move_output_gm(input_gm_addr, output_gm_addr, rows, [ub1, ub2])

        with inst.if_scope(tik.all(out_loops_idx == out_loops - 1, output_matrix_count % self.block_element != 0)):
            floor_align_count = output_matrix_count // self.block_element * self.block_element
            self._data_move(self.output_gm[output_gm_addr], ub2, floor_align_count)
            with inst.for_range(0, self.block_element, name="block_element_loop") as element_id:
                ub_block[element_id] = ub2[output_matrix_count - self.block_element + element_id]
            self._data_move(self.output_gm[output_gm_addr + output_matrix_count -
                                            self.block_element], ub_block, self.block_element)

        with inst.else_scope():
            self._data_move(self.output_gm[output_gm_addr], ub2, output_matrix_count)


# pylint: disable=locally-disabled,too-many-arguments,
# pylint: disable=unused-argument,too-many-locals
@register_operator("StridedSlice")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def strided_slice(input_x, begin, end, strides=None, output_x=None, begin_mask=0, end_mask=0,
                  ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, kernel_name="strided_slice"):
    """
    Extracts a strided slice of a tensor (generalized python array indexing).
    Roughly speaking, this op extracts a slice of size (end-begin)/stride
    from the given input_ tensor.
    Starting at the location specified by begin the slice continues
     by adding stride to the index
    until all dimensions are not less than end. Note that a stride
    can be negative, which causes a reverse slice.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    begin: dict.
        shape and dtype of begin, represents the index of the first value to select.
    end: dict.
        shape and dtype of end, represents the index of the last value to select.
    strides: dict.
        shape and dtype of strides, step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin
        value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position
        is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification
        should shrink the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice"

    Returns
    -------
    tik_instance
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "bool", "int8")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    strided_slice_instance = StridedSlice(input_x, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                          shrink_axis_mask, kernel_name)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = {"out_of_bound_sync_check": True}
    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=(strided_slice_instance.input_gm,
                          strided_slice_instance.begin_gm,
                          strided_slice_instance.end_gm,
                          strided_slice_instance.strides_gm),
                  outputs=(strided_slice_instance.output_gm,),
                  flowtable=[strided_slice_instance.tiling_param.tiling_gm],
                  config=opt_config,
                  enable_l2=False)

    tbe_context.get_context().add_compile_info("vars", {"block_dim": strided_slice_instance.aicore_num,
                                                        "begin_mask": strided_slice_instance.begin_mask,
                                                        "end_mask": strided_slice_instance.end_mask,
                                                        "ellipsis_mask": strided_slice_instance.ellipsis_mask,
                                                        "new_axis_mask": strided_slice_instance.new_axis_mask,
                                                        "shrink_axis_mask": strided_slice_instance.shrink_axis_mask,
                                                        "ub_size": tik.Dprofile().get_unified_buffer_size() - \
                                                                   Tiling_UB_SIZE})
    return inst
