#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
avg_pool
"""
import math
import re
import copy
import warnings
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.dynamic.conv2d import conv2d
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_register


AVG_KERNEL_SIZE_H_MUL_W = 255 #kernel_h * kernel_w
AVG_KERNEL_SIZE = 20 # maximum ksize
MAX_CUBE_STRIDE = 63 # maximum cube stride
NONETYPE = type(None)
MAX_INT32 = 2**31 - 1
TILING_ARG_NUM = 24
RESERVED_UB_SIZE = 8 * 1024
EIGHT_BIT = 8
BLOCK_BYTES = 32
C_ZERO = 16
REPEAT_LIMIT = 255
MASK = 128
MIN_FP16 = 0
INPUT_DIM = 4
SHAPE_LEN = 5
ORI_SHAPE_LEN = 4

class AvgPool:
    def __init__(self, dtype, ksize, strides, padding, kernel_name):
        self.dtype = dtype
        self.ksize_h = ksize[0]
        self.ksize_w = ksize[1]
        self.strides_h = strides[0]
        self.strides_w = strides[1]
        self.padding = 0 if padding == "SAME" else 1
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = tbe_platform.get_bit_len(self.dtype) // EIGHT_BIT
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE) // self.dtype_size
        self.one_fourth_ub_ele = self.ub_ele // 4
        self.init_gm_tensor()
    def tiling_args(self):
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.act_core_num.set_as(self.tiling_ub[1])
        self.one_core_ele = self.tik_instance.Scalar("int32", name="one_core_ele")
        self.one_core_ele.set_as(self.tiling_ub[2])
        self.last_core_ele = self.tik_instance.Scalar("int32", name="last_core_ele")
        self.last_core_ele.set_as(self.tiling_ub[3])
        self.input_h = self.tik_instance.Scalar("int32", name="input_h")
        self.input_h.set_as(self.tiling_ub[4])
        self.input_w = self.tik_instance.Scalar("int32", name="input_w")
        self.input_w.set_as(self.tiling_ub[5])
        self.output_h = self.tik_instance.Scalar("int32", name="output_h")
        self.output_h.set_as(self.tiling_ub[6])
        self.output_w = self.tik_instance.Scalar("int32", name="output_w")
        self.output_w.set_as(self.tiling_ub[7])
        self.pad_h = self.tik_instance.Scalar("int32", name="pad_h")
        self.pad_h.set_as(self.tiling_ub[8])
        self.pad_w = self.tik_instance.Scalar("int32", name="pad_w")
        self.pad_w.set_as(self.tiling_ub[9])
        self.pad_t = self.tik_instance.Scalar("int32", name="pad_t")
        self.pad_t.set_as(self.tiling_ub[10])
        self.pad_b = self.tik_instance.Scalar("int32", name="pad_b")
        self.pad_b.set_as(self.tiling_ub[11])
        self.pad_l = self.tik_instance.Scalar("int32", name="pad_l")
        self.pad_l.set_as(self.tiling_ub[12])
        self.pad_r = self.tik_instance.Scalar("int32", name="pad_r")
        self.pad_r.set_as(self.tiling_ub[13])
        self.c_factor = self.tik_instance.Scalar("int32", name="c_factor")
        self.c_factor.set_as(self.tiling_ub[14])
        self.h_factor = self.tik_instance.Scalar("int32", name="h_factor")
        self.h_factor.set_as(self.tiling_ub[15])
        self.w_factor = self.tik_instance.Scalar("int32", name="w_factor")
        self.w_factor.set_as(self.tiling_ub[16])
        self.one_core_loop_num = self.tik_instance.Scalar("int32", name="one_core_loop_num")
        self.one_core_loop_num.set_as(self.tiling_ub[17])
        self.one_core_loop_left = self.tik_instance.Scalar("int32", name="one_core_loop_left")
        self.one_core_loop_left.set_as(self.tiling_ub[18])
        self.last_core_loop_num = self.tik_instance.Scalar("int32", name="last_core_loop_num")
        self.last_core_loop_num.set_as(self.tiling_ub[19])
        self.last_core_loop_left = self.tik_instance.Scalar("int32", name="last_core_loop_left")
        self.last_core_loop_left.set_as(self.tiling_ub[20])

    def init_gm_tensor(self):
        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="output_gm", scope=tik.scope_gm)

    def init_ub_tensor(self):
        self.ub_a = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_a", scope=tik.scope_ubuf)
        self.ub_b = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_b", scope=tik.scope_ubuf)
        self.ub_c = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_c", scope=tik.scope_ubuf)
        self.ub_d = self.tik_instance.Tensor(self.dtype, (self.one_fourth_ub_ele,), name="ub_d", scope=tik.scope_ubuf)

    def init_ub_scalar_single(self):
        self.size = self.tik_instance.Scalar("int32", name="size")
        self.offset = self.tik_instance.Scalar("int32", name="offset")
        self.core_ele = self.tik_instance.Scalar("int32", name="core_ele")
        self.loop_num = self.tik_instance.Scalar("int32", name="loop_num")
        self.loop_left = self.tik_instance.Scalar("int32", name="loop_left")
        self.before_h = self.tik_instance.Scalar("int32", name="before_h")
        self.after_h = self.tik_instance.Scalar("int32", name="after_h")
        self.before_w = self.tik_instance.Scalar("int32", name="before_w")
        self.after_w = self.tik_instance.Scalar("int32", name="after_w")
        self.len_h = self.tik_instance.Scalar("int32", name="len_h")
        self.len_w = self.tik_instance.Scalar("int32", name="len_w")
        self.offset_ub = self.tik_instance.Scalar("int32", name="offset_ub")
        self.nburst = self.tik_instance.Scalar("int32", name="nburst")
        self.src_stride = self.tik_instance.Scalar("int32", name="src_stride")
        self.dst_stride = self.tik_instance.Scalar("int32", name="dst_stride")
        self.burst_len_in = self.tik_instance.Scalar("int32", name="burst_len_in")
        self.burst_len_out = self.tik_instance.Scalar("int32", name="burst_len_out")
        self.size_1 = self.tik_instance.Scalar("int32", name="size_1")
        self.offset_2 = self.tik_instance.Scalar("int32", name="offset_2")
        self.size_2 = self.tik_instance.Scalar("int32", name="size_2")
        self.offset_3 = self.tik_instance.Scalar("int32", name="offset_3")
        self.repeat_3 = self.tik_instance.Scalar("int32", name="repeat_3")
        self.size_3 = self.tik_instance.Scalar("int32", name="size_3")
        self.size_w = self.tik_instance.Scalar("int32", name="size_w")
        self.repeat_w = self.tik_instance.Scalar("int32", name="repeat_w")
        self.rep_blk_h = self.tik_instance.Scalar("int32", name="rep_blk_h")
        self.factor_total = self.tik_instance.Scalar(self.dtype, name="factor_total")

    def init_ub_scalar_double(self):
        self.size_loop = self.tik_instance.Scalar("int32", name="size_loop")
        self.size_left = self.tik_instance.Scalar("int32", name="size_left")
        self.repeat_loop = self.tik_instance.Scalar("int32", name="repeat_loop")
        self.repeat_left = self.tik_instance.Scalar("int32", name="repeat_left")
        self.size_offset = self.tik_instance.Scalar("int32", name="size_offset")
        self.repeat_offset = self.tik_instance.Scalar("int32", name="repeat_offset")
        self.offset_in = self.tik_instance.Scalar("int32", name="offset_in")
        self.offset_out = self.tik_instance.Scalar("int32", name="offset_out")
        self.repeat_offset_src0 = self.tik_instance.Scalar("int32", name="repeat_offset_src0")
        self.repeat_offset_src1 = self.tik_instance.Scalar("int32", name="repeat_offset_src1")
        self.size_offset_src0 = self.tik_instance.Scalar("int32", name="size_offset_src0")
        self.size_offset_src1 = self.tik_instance.Scalar("int32", name="size_offset_src1")
        self.offset_dst = self.tik_instance.Scalar("int32", name="offset_dst")
        self.offset_src0 = self.tik_instance.Scalar("int32", name="offset_src0")
        self.offset_src1 = self.tik_instance.Scalar("int32", name="offset_src1")
        self.offset_h = self.tik_instance.Scalar("int32", name="offset_h")
        self.offset_w = self.tik_instance.Scalar("int32", name="offset_w")

    def select_tiling_mode(self, core_idx, core_ele, loop_num, loop_left):
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self.tiling_c_dim_core_nc(core_idx, core_ele)
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            self.tiling_h_dim_core_nc(core_idx, core_ele, loop_num, loop_left)
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.tiling_w_dim_core_nc(core_idx, core_ele, loop_num, loop_left)

    def vector_dup_continuous(self, src, size):
        with self.tik_instance.if_scope(size > 0):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(self.size_loop // REPEAT_LIMIT)
            self.repeat_left.set_as(self.size_loop % REPEAT_LIMIT)
            with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * MASK)
                self.tik_instance.vector_dup(MASK, src[self.repeat_offset], MIN_FP16, REPEAT_LIMIT, 1, 8)
            with self.tik_instance.if_scope(self.repeat_left > 0):
                self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * MASK)
                self.tik_instance.vector_dup(MASK, src[self.repeat_offset], MIN_FP16, self.repeat_left, 1, 8)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                self.tik_instance.vector_dup(self.size_left, src[self.size_offset], MIN_FP16, 1, 1, 8)

    def vector_dup_discrete(self, src, repeat, size, dst_blk=1, dst_rep=8):
        with self.tik_instance.if_scope(tik.all(size > 0, repeat > 0)):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(repeat // REPEAT_LIMIT)
            self.repeat_left.set_as(repeat % REPEAT_LIMIT)
            def _inner(src, mask_len):
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.tik_instance.vector_dup(mask_len, src[self.repeat_offset], MIN_FP16, REPEAT_LIMIT, dst_blk,
                                                 dst_rep)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.tik_instance.vector_dup(mask_len, src[self.repeat_offset], MIN_FP16, self.repeat_left, dst_blk,
                                                 dst_rep)

            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.size_offset.set_as(size_loop_idx * MASK)
                _inner(src[self.size_offset:], MASK)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                _inner(src[self.size_offset:], self.size_left)

    def reduce_max_rw(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        if self.strides_w <= 31:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // MASK)
                self.size_left.set_as(size % MASK)
                self.repeat_loop.set_as(self.size_loop // REPEAT_LIMIT)
                self.repeat_left.set_as(self.size_loop % REPEAT_LIMIT)
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vmax(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vmax(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.size_left > 0):
                    self.size_offset.set_as(self.size_loop * MASK)
                    self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                    self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                    self.tik_instance.vmax(self.size_left, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
        else:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // C_ZERO)
                with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                    self.size_offset.set_as(size_loop_idx * C_ZERO)
                    self.size_offset_src0.set_as(size_loop_idx * src0_blk * C_ZERO)
                    self.size_offset_src1.set_as(size_loop_idx * src1_blk * C_ZERO)
                    self.tik_instance.vmax(C_ZERO, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, 1, 1, 1, 1, 1)

    def reduce_avg_rw(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        if self.strides_w <= 31:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // MASK)
                self.size_left.set_as(size % MASK)
                self.repeat_loop.set_as(self.size_loop // REPEAT_LIMIT)
                self.repeat_left.set_as(self.size_loop % REPEAT_LIMIT)
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vadd(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * MASK)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_blk * MASK)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_blk * MASK)
                    self.tik_instance.vadd(MASK, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
                with self.tik_instance.if_scope(self.size_left > 0):
                    self.size_offset.set_as(self.size_loop * MASK)
                    self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                    self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                    self.tik_instance.vadd(self.size_left, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, src0_blk, src1_blk, 8,
                                           src0_blk * 8, src1_blk * 8)
        else:
            with self.tik_instance.if_scope(size > 0):
                self.size_loop.set_as(size // C_ZERO)
                with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                    self.size_offset.set_as(size_loop_idx * C_ZERO)
                    self.size_offset_src0.set_as(size_loop_idx * src0_blk * C_ZERO)
                    self.size_offset_src1.set_as(size_loop_idx * src1_blk * C_ZERO)
                    self.tik_instance.vadd(C_ZERO, dst[self.size_offset], src0[self.size_offset_src0],
                                           src1[self.size_offset_src1], 1, 1, 1, 1, 1, 1, 1)

    def reduce_max_rh(self, dst, src0, src1, repeat, size, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8, src1_rep=8):
        with self.tik_instance.if_scope(tik.all(size > 0, repeat > 0)):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(repeat // REPEAT_LIMIT)
            self.repeat_left.set_as(repeat % REPEAT_LIMIT)
            def _inner(dst, src0, src1, mask_len):
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vmax(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, dst_rep,
                                           src0_rep, src1_rep)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vmax(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk,
                                           dst_rep, src0_rep, src1_rep)

            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.size_offset.set_as(size_loop_idx * MASK)
                self.size_offset_src0.set_as(size_loop_idx * src0_blk * MASK)
                self.size_offset_src1.set_as(size_loop_idx * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:], MASK)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:],
                       self.size_left)

    def reduce_avg_rh(self, dst, src0, src1, repeat, size, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8, src1_rep=8):
        with self.tik_instance.if_scope(tik.all(size > 0, repeat > 0)):
            self.size_loop.set_as(size // MASK)
            self.size_left.set_as(size % MASK)
            self.repeat_loop.set_as(repeat // REPEAT_LIMIT)
            self.repeat_left.set_as(repeat % REPEAT_LIMIT)
            def _inner(dst, src0, src1, mask_len):
                with self.tik_instance.for_range(0, self.repeat_loop) as repeat_loop_idx:
                    self.repeat_offset.set_as(repeat_loop_idx * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(repeat_loop_idx * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(repeat_loop_idx * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vadd(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], REPEAT_LIMIT, 1, src0_blk, src1_blk, dst_rep,
                                           src0_rep, src1_rep)
                with self.tik_instance.if_scope(self.repeat_left > 0):
                    self.repeat_offset.set_as(self.repeat_loop * REPEAT_LIMIT * dst_rep * C_ZERO)
                    self.repeat_offset_src0.set_as(self.repeat_loop * REPEAT_LIMIT * src0_rep * C_ZERO)
                    self.repeat_offset_src1.set_as(self.repeat_loop * REPEAT_LIMIT * src1_rep * C_ZERO)
                    self.tik_instance.vadd(mask_len, dst[self.repeat_offset], src0[self.repeat_offset_src0],
                                           src1[self.repeat_offset_src1], self.repeat_left, 1, src0_blk, src1_blk,
                                           dst_rep, src0_rep, src1_rep)

            with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
                self.size_offset.set_as(size_loop_idx * MASK)
                self.size_offset_src0.set_as(size_loop_idx * src0_blk * MASK)
                self.size_offset_src1.set_as(size_loop_idx * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:], MASK)
            with self.tik_instance.if_scope(self.size_left > 0):
                self.size_offset.set_as(self.size_loop * MASK)
                self.size_offset_src0.set_as(self.size_loop * src0_blk * MASK)
                self.size_offset_src1.set_as(self.size_loop * src1_blk * MASK)
                _inner(dst[self.size_offset:], src0[self.size_offset_src0:], src1[self.size_offset_src1:],
                       self.size_left)

    def reduce_max(self, ub_x, ub_y, repeat_p, repeat_o):
        if self.ksize_w == 1:
            if self.strides_w == self.ksize_w:
                self.reduce_max_rw(ub_y, ub_x, ub_x, repeat_p * self.size_w, self.strides_w, self.strides_w)
            elif self.strides_w <= 255:
                with self.tik_instance.if_scope(
                        tik.all(repeat_p >= self.repeat_w, self.output_w <= 255, self.pad_w <=255)):
                    self.reduce_max_rh(ub_y, ub_x, ub_x, repeat_p, self.size_w, self.strides_w, self.strides_w,
                                       self.output_w, self.pad_w, self.pad_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_p) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                        self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src0:],
                                           self.size_w, self.strides_w, self.strides_w)
            else:
                with self.tik_instance.for_range(0, repeat_p) as h_idx:
                    self.offset_dst.set_as(h_idx * self.size_w)
                    self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                    self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src0:],
                                        self.size_w, self.strides_w, self.strides_w)
        else:
            if self.strides_w == self.ksize_w:
                self.reduce_avg_rw(ub_y, ub_x, ub_x[C_ZERO:], repeat_p * self.size_w, self.strides_w, self.strides_w)
                with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                    self.offset_w.set_as((idx + 2) * C_ZERO)
                    self.reduce_avg_rw(ub_y, ub_x[self.offset_w:], ub_y, repeat_p * self.size_w, self.strides_w, 1)
            elif self.strides_w <= 255:
                with self.tik_instance.if_scope(
                        tik.all(repeat_p >= self.repeat_w, self.output_w <= 255, self.pad_w <=255)):
                    self.reduce_avg_rh(ub_y, ub_x, ub_x[C_ZERO:], repeat_p, self.size_w, self.strides_w, self.strides_w,
                                       self.output_w, self.pad_w, self.pad_w)
                    with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                        self.offset_w.set_as((idx + 2) * C_ZERO)
                        self.reduce_avg_rh(ub_y, ub_x[self.offset_w:], ub_y, repeat_p, self.size_w, self.strides_w, 1,
                                       self.output_w, self.pad_w, self.output_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_p) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                        self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                        self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src1:],
                                           self.size_w, self.strides_w, self.strides_w)
                        with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                            self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                            self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:], ub_y[self.offset_dst:],
                                               self.size_w, self.strides_w, 1)
            else:
                with self.tik_instance.for_range(0, repeat_p) as h_idx:
                    self.offset_dst.set_as(h_idx * self.size_w)
                    self.offset_src0.set_as(h_idx * self.pad_w * C_ZERO)
                    self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                    self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src1:],
                                        self.size_w, self.strides_w, self.strides_w)
                    with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                        self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                        self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:], ub_y[self.offset_dst:],
                                            self.size_w, self.strides_w, 1)

        if self.ksize_h == 1:
            if self.strides_h == 1:
                self.reduce_max_rw(ub_x, ub_y, ub_y, repeat_o * self.size_w, 1, 1)
            else:
                with self.tik_instance.if_scope(
                        tik.all(repeat_o >= self.repeat_w, self.output_w <= 255, self.rep_blk_h <=255)):
                    self.reduce_max_rh(ub_x, ub_y, ub_y, repeat_o, self.size_w, 1, 1,
                                       self.output_w, self.rep_blk_h, self.rep_blk_h)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_o) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.strides_h * self.size_w)
                        self.reduce_max_rw(ub_x[self.offset_dst:], ub_y[self.offset_src0:], ub_y[self.offset_src0:],
                                           self.size_w, 1, 1)
        else:
            if self.strides_h == 1:
                self.reduce_avg_rw(ub_x, ub_y, ub_y[self.size_w:], repeat_o * self.size_w, 1, 1)
                with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                    self.offset_h.set_as((idx + 2) * self.size_w)
                    self.reduce_avg_rw(ub_x, ub_y[self.offset_h:], ub_x, repeat_o * self.size_w, 1, 1)
            else:
                with self.tik_instance.if_scope(
                        tik.all(repeat_o >= self.repeat_w, self.output_w <= 255, self.rep_blk_h <=255)):
                    self.reduce_avg_rh(ub_x, ub_y, ub_y[self.size_w:], repeat_o, self.size_w, 1, 1,
                                       self.output_w, self.rep_blk_h, self.rep_blk_h)
                    with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                        self.offset_h.set_as((idx + 2) * self.size_w)
                        self.reduce_avg_rh(ub_x, ub_y[self.offset_h:], ub_y, repeat_o, self.size_w, 1, 1,
                                       self.output_w, self.rep_blk_h, self.output_w)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, repeat_o) as h_idx:
                        self.offset_dst.set_as(h_idx * self.size_w)
                        self.offset_src0.set_as(h_idx * self.strides_h * self.size_w)
                        self.offset_src1.set_as(self.offset_src0 + self.size_w)
                        self.reduce_avg_rw(ub_x[self.offset_dst:], ub_y[self.offset_src0:], ub_y[self.offset_src1:],
                                           self.size_w, 1, 1)
                        with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                            self.offset_h.set_as(self.offset_src0 + (idx + 2) * self.size_w)
                            self.reduce_avg_rw(ub_x[self.offset_dst:], ub_y[self.offset_h:], ub_x[self.offset_dst:],
                                               self.size_w, 1, 1)
        self.factor_total.set_as(1.0 / (self.ksize_h * self.ksize_w))
        self.size_loop.set_as(repeat_o * self.size_w // C_ZERO)
        with self.tik_instance.for_range(0, self.size_loop) as size_loop_idx:
            self.size_offset.set_as(size_loop_idx * C_ZERO)
            self.tik_instance.vmuls(C_ZERO, ub_x[self.size_offset], ub_x[self.size_offset], self.factor_total, 1, 1, 1, 1, 1)

    def tiling_c_dim_core_nc(self, core_idx, core_ele):
        '''Tiling c1 dim
        '''
        #common params
        self.size_1.set_as(self.pad_t * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
        self.offset_2.set_as((self.pad_h - self.pad_b) * self.pad_w *C_ZERO - self.pad_r * C_ZERO)
        self.size_2.set_as(self.pad_b * self.pad_w * C_ZERO + self.pad_r * C_ZERO)
        self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
        self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
        self.size_w.set_as(self.output_w * C_ZERO)
        self.burst_len_out.set_as(self.output_h * self.output_w)
        #
        self.size.set_as(self.pad_h * self.pad_w * C_ZERO)
        #
        with self.tik_instance.if_scope(self.pad_h <= self.input_h):
            self.nburst.set_as(self.pad_h)
        with self.tik_instance.else_scope():
            self.nburst.set_as(self.input_h)
        #
        with self.tik_instance.if_scope(self.pad_w <= self.input_w):
            self.repeat_3.set_as(0)
            self.burst_len_in.set_as(self.pad_w)
            self.src_stride.set_as(self.input_w - self.pad_w)
            self.dst_stride.set_as(0)
        with self.tik_instance.else_scope():
            self.repeat_3.set_as(self.pad_h -1)
            self.burst_len_in.set_as(self.input_w)
            self.src_stride.set_as(0)
            self.dst_stride.set_as(self.pad_r + self.pad_l)

        #
        with self.tik_instance.new_stmt_scope():

            def _inner(ele_idx, ub_x, ub_y):
                self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w  * C_ZERO)
                self.offset_out.set_as(
                    (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w  * C_ZERO)
                #
                with self.tik_instance.if_scope(tik.all(self.pad_w > 255, self.size_3 > 0)):
                    self.vector_dup_continuous(ub_x, self.size)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
                self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[self.offset_in], 0, self.nburst,
                                            self.burst_len_in,self.src_stride, self.dst_stride)
                #
                self.reduce_max(ub_x, ub_y, self.pad_h, self.output_h)
                #
                self.tik_instance.data_move(self.output_gm[self.offset_out],ub_x, 0, 1, self.burst_len_out,0 , 0)
            
            #
            self.init_ub_tensor()
            self.init_ub_scalar_double()
            with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                _inner(ele_idx * 2, self.ub_a, self.ub_b)
                _inner(ele_idx * 2 +1,  self.ub_c, self.ub_d)
            with self.tik_instance.if_scope(core_ele % 2 == 1):
                _inner(core_ele -1,  self.ub_a, self.ub_b)

    def tiling_h_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_idx, self.h_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_h_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_h_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        #common params
        self.before_h.set_as(loop_idx * self.h_factor * self.strides_h)
        self.after_h.set_as((loop_idx * self.h_factor +ele -1) * self.strides_h + self.ksize_h)
        self.len_h.set_as(self.after_h - self.before_h)
        self.size_w.set_as(self.output_w * C_ZERO)
        self.burst_len_out.set_as(ele * self.output_w)
        #
        self.size.set_as(self.len_h * self.pad_w * C_ZERO)
        #
        with self.tik_instance.if_scope(self.before_h < self.pad_t):
            self.size_1.set_as((self.pad_t -self.before_h) * self.pad_w * C_ZERO + self.pad_l * C_ZERO)
            self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
            self.offset.set_as(0)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h -self.pad_b):
                self.offset_2.set_as(self.len_h * self.pad_w *C_ZERO - self.pad_r * C_ZERO)
                self.size_2.set_as(self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.after_h - self.pad_t -1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.after_h - self.pad_t)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h -self.pad_b - self.before_h) * self.pad_w *C_ZERO -
                                    self.pad_r * C_ZERO)
                self.size_2.set_as((self.after_h -(self.pad_h - self.pad_b)) * self.pad_w *C_ZERO +
                                  self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.input_h-1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.input_h)
        with self.tik_instance.else_scope():
            self.size_1.set_as(self.pad_l * C_ZERO)
            self.offset_3.set_as(self.size_1 + self.input_w * C_ZERO)
            self.offset.set_as((self.before_h - self.pad_t) * self.input_w * C_ZERO)
            with self.tik_instance.if_scope(self.after_h <= self.pad_h -self.pad_b):
                self.offset_2.set_as(self.len_h * self.pad_w *C_ZERO - self.pad_r * C_ZERO)
                self.size_2.set_as(self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.len_h-1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.len_h)
            with self.tik_instance.else_scope():
                self.offset_2.set_as((self.pad_h -self.pad_b - self.before_h) * self.pad_w *C_ZERO -
                                    self.pad_r * C_ZERO)
                self.size_2.set_as((self.after_h -(self.pad_h - self.pad_b)) * self.pad_w *C_ZERO +
                                  self.pad_r * C_ZERO)
                self.repeat_3.set_as(self.pad_h - self.pad_b - self.before_h - 1)
                self.size_3.set_as((self.pad_r + self.pad_l) * C_ZERO)
                self.nburst.set_as(self.pad_h - self.pad_b - self.before_h)
        #
        with self.tik_instance.if_scope(self.pad_w <= self.input_w):
            self.burst_len_in.set_as(self.pad_w)
            self.src_stride.set_as(self.input_w - self.pad_w)
            self.dst_stride.set_as(0)
        with self.tik_instance.else_scope():
            self.burst_len_in.set_as(self.input_w)
            self.src_stride.set_as(0)
            self.dst_stride.set_as(self.pad_r + self.pad_l)

        #
        with self.tik_instance.new_stmt_scope():

            def _inner(ele_idx, ub_x, ub_y):
                self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w  * C_ZERO +
                                      self.offset)
                self.offset_out.set_as((core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w  *
                                        C_ZERO + loop_idx * self.h_factor * self.output_w * C_ZERO)
                #
                with self.tik_instance.if_scope(tik.all(self.pad_w > 255, self.size_3 > 0)):
                    self.vector_dup_continuous(ub_x, self.size)
                with self.tik_instance.else_scope():
                    self.vector_dup_continuous(ub_x, self.size_1)
                    self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                    self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.pad_w)
                self.tik_instance.data_move(ub_x[self.size_1], self.input_gm[self.offset_in], 0, self.nburst,
                                            self.burst_len_in,self.src_stride, self.dst_stride)
                #
                self.reduce_max(ub_x, ub_y, self.len_h, ele)
                #
                self.tik_instance.data_move(self.output_gm[self.offset_out],ub_x, 0, 1, self.burst_len_out,0 , 0)
            
            #
            self.init_ub_tensor()
            self.init_ub_scalar_double()
            with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                _inner(ele_idx * 2, self.ub_a, self.ub_b)
                _inner(ele_idx * 2 +1,  self.ub_c, self.ub_d)
            with self.tik_instance.if_scope(core_ele % 2 == 1):
                _inner(core_ele -1,  self.ub_a, self.ub_b)

    def tiling_w_dim_core_nc(self, core_idx, core_ele, loop_num, loop_left):
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.tiling_w_dim_core_nc_process(core_idx, core_ele, loop_idx, self.w_factor)
        with self.tik_instance.if_scope(loop_left > 0):
            self.tiling_w_dim_core_nc_process(core_idx, core_ele, loop_num, loop_left)

    def tiling_w_dim_core_nc_process(self, core_idx, core_ele, loop_idx, ele):
        #common params
        self.before_w.set_as(loop_idx * self.w_factor * self.strides_w)
        self.after_w.set_as((loop_idx * self.w_factor +ele -1) * self.strides_w + self.ksize_w)
        self.len_w.set_as(self.after_w - self.before_w)
        self.size_w.set_as(ele * C_ZERO)
        self.burst_len_out.set_as(ele)
        #
        self.size.set_as(self.ksize_h * self.len_w * C_ZERO)
        #
        with self.tik_instance.for_range(0, self.output_h) as h_idx:
            #
            self.before_h.set_as(h_idx * self.h_factor * self.strides_h)
            self.after_h = self.before_h + self.ksize_h
            with self.tik_instance.if_scope(self.before_h < self.pad_t):
                #
                self.size_1.set_as((self.pad_t - self.before_h) * self.len_w* C_ZERO)
                self.offset_2.set_as(0)
                self.size_2.set_as(0)
                self.repeat_3.set_as(self.after_h - self.pad_t)
                self.nburst.set_as(self.after_h - self.pad_t)
                #
                with self.tik_instance.if_scope(self.before_w < self.pad_l):
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(self.size_1)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset.set_as(0)
                        self.offset_ub.set_as(self.size_1 + (self.pad_l- self.before_w) * C_ZERO)
                        self.burst_len_in.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w- self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset.set_as((self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(self.size_1)
                    with self.tik_instance.if_scope(self.after_w <= self.pad_w -self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len_in.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as(self.size_1 + (self.pad_w - self.pad_r - self.before_w)* C_ZERO)
                        self.size_3.set_as((self.after_w - (self.pad_w - self.pad_r))* C_ZERO)
                        self.burst_len_in.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))
            with self.tik_instance.else_scope():
                #
                self.size_1.set_as(0)
                with self.tik_instance.if_scope(self.after_h <= self.pad_h -self.pad_b):
                    self.offset_2.set_as(0)
                    self.size_2.set_as(0)
                    self.repeat_3.set_as(self.ksize_h)
                    self.nburst.set_as(self.ksize_h)
                with self.tik_instance.else_scope():
                    self.offset_2.set_as((self.pad_h - self.pad_b - self.before_h) * self.len_w * C_ZERO)
                    self.size_2.set_as((self.after_h - (self.pad_h - self.pad_b)) * self.len_w * C_ZERO)
                    self.repeat_3.set_as(self.pad_h - self.pad_b - self.before_h)
                    self.nburst.set_as(self.pad_h - self.pad_b - self.before_h)
                #
                with self.tik_instance.if_scope(self.before_w  < self.pad_l):    
                    with self.tik_instance.if_scope(self.after_w  <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as((self.pad_l - self.before_w) * C_ZERO)
                        self.offset.set_as((self.before_h - self.pad_t) * self.input_w* C_ZERO)
                        self.offset_ub.set_as(self.size_3)
                        self.burst_len_in.set_as(self.after_w - self.pad_l)
                        self.src_stride.set_as(self.input_w - (self.after_w - self.pad_l))
                        self.dst_stride.set_as(self.pad_l - self.before_w)
                with self.tik_instance.else_scope():
                    self.offset.set_as((self.before_h - self.pad_t) * self.input_w* C_ZERO +
                                        (self.before_w - self.pad_l) * C_ZERO)
                    self.offset_ub.set_as(0)
                    with self.tik_instance.if_scope(self.after_w  <= self.pad_w - self.pad_r):
                        self.offset_3.set_as(0)
                        self.size_3.set_as(0)
                        self.burst_len_in.set_as(self.len_w)
                        self.src_stride.set_as(self.input_w - self.len_w)
                        self.dst_stride.set_as(0)
                    with self.tik_instance.else_scope():
                        self.offset_3.set_as((self.pad_w - self.pad_r - self.before_w)* C_ZERO)
                        self.size_3.set_as(self.after_w - (self.pad_w - self.pad_r)* C_ZERO)
                        self.burst_len_in.set_as(self.pad_w - self.pad_r - self.before_w)
                        self.src_stride.set_as(self.before_w - self.pad_l)
                        self.dst_stride.set_as(self.after_w - (self.pad_w - self.pad_r))

            with self.tik_instance.new_stmt_scope():
                def _inner(ele_idx, ub_x, ub_y):
                    self.offset_in.set_as((core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w *
                                          C_ZERO + self.offset)
                    self.offset_out.set_as((core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w *
                                          C_ZERO + h_idx * self.output_w * C_ZERO + loop_idx *self.w_factor * C_ZERO)
                    with self.tik_instance.if_scope(tik.all(self.len_w > 255, self.size_3 > 0)):
                        self.vector_dup_continuous(ub_x, self.size)
                    with self.tik_instance.else_scope():
                        self.vector_dup_continuous(ub_x, self.size_1)
                        self.vector_dup_continuous(ub_x[self.offset_2:], self.size_2)
                        self.vector_dup_discrete(ub_x[self.offset_3:], self.repeat_3, self.size_3, 1, self.len_w)
                    self.tik_instance.data_move(ub_x[self.offset_ub], self.input_gm[self.offset_in], 0, self.nburst,
                                                self.burst_len_in, self.src_stride, self.dst_stride)
                    if self.ksize_w == 1:
                        with self.tik_instance.for_range(0, self.ksize_h) as k_idx:
                            self.offset_dst.set_as(k_idx * self.size_w)
                            self.offset_src0.set_as(k_idx * self.len_w * C_ZERO)
                            self.reduce_max_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src0:],
                                               self.size_w, self.strides_w, self.strides_w)
                    else:
                        with self.tik_instance.for_range(0, self.ksize_h) as k_idx:
                            self.offset_dst.set_as(k_idx * self.size_w)
                            self.offset_src0.set_as(k_idx * self.len_w * C_ZERO)
                            self.offset_src1.set_as(self.offset_src0 + C_ZERO)
                            self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_src0:], ub_x[self.offset_src1:],
                                               self.size_w, self.strides_w, self.strides_w)
                            with self.tik_instance.for_range(0, self.ksize_w - 2) as idx:
                                self.offset_w.set_as(self.offset_src0 + (idx + 2) * C_ZERO)
                                self.reduce_avg_rw(ub_y[self.offset_dst:], ub_x[self.offset_w:], ub_y[self.offset_dst:],
                                               self.size_w, self.strides_w, 1)
                    if self.ksize_h == 1:
                        self.reduce_max_rw(ub_x, ub_y, ub_y, self.size_w, 1, 1)
                    else:
                        self.reduce_avg_rw(ub_x, ub_y, ub_y[self.size_w:], self.size_w, 1, 1)
                        with self.tik_instance.for_range(0, self.ksize_h - 2) as idx:
                            self.offset_h.set_as((idx + 2) * self.size_w)
                            self.reduce_avg_rw(ub_x, ub_y[self.offset_h:], ub_x, self.size_w, 1, 1)
                    self.tik_instance.data_move(self.output_gm[self.offset_out], ub_x, 0, 1, self.burst_len_out, 0, 0)
                self.init_ub_tensor()
                self.init_ub_scalar_double()
                with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                    _inner(ele_idx * 2, self.ub_a, self.ub_b)
                    _inner(ele_idx * 2 + 1, self.ub_c, self.ub_d)
                with self.tik_instance.if_scope(core_ele % 2 == 1):
                    _inner(core_ele - 1, self.ub_a, self.ub_b)

    def avg_pool_compute_tiling(self):
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_idx:
            self.tiling_ub = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,),
                                                      name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)
            self.tiling_args()
            self.init_ub_scalar_single()
            with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                self.core_ele.set_as(self.one_core_ele)
                self.loop_num.set_as(self.one_core_loop_num)
                self.loop_left.set_as(self.one_core_loop_left)
            with self.tik_instance.if_scope(core_idx == self.act_core_num - 1):
                self.core_ele.set_as(self.last_core_ele)
                self.loop_num.set_as(self.last_core_loop_num)
                self.loop_left.set_as(self.last_core_loop_left)
            self.select_tiling_mode(core_idx, self.core_ele, self.loop_num, self.loop_left)

    def avg_pool_operator(self):
        self.avg_pool_compute_tiling()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm])
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": self.ub_ele,
                "core_num": self.core_num,
                "ksize_h": self.ksize_h,
                "ksize_w": self.ksize_w,
                "strides_h": self.strides_h,
                "strides_w": self.strides_w,
                "padding": self.padding
            }
        )
        return self.tik_instance


def correct_fuzz_build_range(x, filter, strides, padding):
    """
    get input w range within L1 size

    Notice
    ------
    the proper range are not smaller then shape value

    Parameters
    ----------
    same to avg_pool

    Returns
    -------
    None
    """
    valid = isinstance(x.get("range"), (list, tuple)) and len(x.get("range")) >= 4
    if not valid:
        return
    proper_range = list(map(list, x["range"]))
    pos_h = 2
    pos_w = 3
    if x["ori_format"] == "NHWC":
        pos_h = 1
        pos_w = 2
    pos_range_h = pos_h
    pos_range_w = pos_w
    if len(proper_range) > 4:
        pos_range_h = 2
        pos_range_w = 3
    # -1 shape range is set by user, should not change
    invaild = (None in proper_range[pos_range_h]) \
              or (None in proper_range[pos_range_w]) \
              or x["ori_shape"][pos_w] == -1
    if invaild:
        return
    # >>> start: get the max proper w right range, at least same to shape value
    type_byte = int(re.findall(r'\d+', x["dtype"])[0]) // 8
    filter_h = filter["ori_shape"][filter["ori_format"].find("H")]
    filter_w = filter["ori_shape"][filter["ori_format"].find("W")]
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")
    proper_w = proper_range[pos_range_w][1]
    for i in list(range(proper_w, x["ori_shape"][pos_w] - 1, -1)):
        if padding == "SAME":
            w_out = i + strides[pos_w] - 1 // strides[pos_w]
        else:
            w_out = (i - filter_w) // strides[pos_w] + 1
        limit_h_out = math.floor(tbe_platform.CUBE_MKN[x["dtype"]]['mac'][0] / w_out) + 2
        limit_size = ((limit_h_out - 1) * strides[pos_h] + filter_h) * i
        limit_size = limit_size * tbe_platform.CUBE_MKN[x["dtype"]]['mac'][1] * type_byte
        proper_w = i
        if limit_size < l1_size:
            break
    # <<< end: get the max proper w right range, at least same to shape value
    # >>> start: change input range and ori_range if exists
    if proper_w != proper_range[pos_range_w][1]:
        proper_range[pos_range_w][1] = proper_w
        to_print = "avg_pool fuzz build range changed from {} to {}".format(x["range"], proper_range)
        x["range"] = proper_range
        valid = isinstance(x.get("ori_range"), (list, tuple)) and len(x["ori_range"]) == 4
        if valid:
            ori_range = list(map(list, x["ori_range"]))
            ori_range[pos_w][1] = proper_w
            to_print = "{}, ori_range changed from {} to {}".format(to_print, x["ori_range"], proper_range)
            x["ori_range"] = ori_range
        warnings.warn(to_print)
    # <<< end: change input range and ori_range if exists


@tbe_register.register_param_generalization("AvgPool")
def avg_pool_generalization(x, filter, bias, y, ksize, strides, padding="VALID",
                            data_format='NHWC', offset_x=0, kernel_name="avg_pool", generalize_config=None):
    """
    avg_pool generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to avg_pool

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    support_mode = ["keep_rank"]
    if generalize_config.get("mode") not in support_mode:
        error_manager_cube.raise_err_specific_user("avg_pool", "invalid generalize mode {}, only support {}".format(
            str(generalize_config.get("mode")), str(support_mode)))
    result = []
    # unknow_rank inputs ori_shape is [-2], others' shape length is 4
    unknow_rank = len(x["ori_shape"]) == 1 and x["ori_shape"][0] == -2
    if unknow_rank:
        error_manager_cube.raise_err_specific_user("avg_pool", "not support unknow_rank under mode {}".format(
            generalize_config["mode"]))
    correct_fuzz_build_range(x, filter, strides, padding)
    have_range = {"x": x, "y": y}
    for name, tensor in have_range.items():
        # only change shape NHW dim to -1, range is already set at infershape
        valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
        if not valid:
            error_manager_cube.raise_err_specific_user("avg_pool", "invalid {} ori_shape {}, only support {}d".format(
                    name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
        valid = isinstance(tensor.get("shape"), (list, tuple)) and len(tensor["shape"]) == SHAPE_LEN
        if not valid:
            error_manager_cube.raise_err_specific_user("avg_pool", "invalid {} ori_shape {}, only support {}d".format(
                    name, str(tensor.get("shape")), str(SHAPE_LEN)))
        tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
            if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
        tensor["shape"] = [-1, tensor["shape"][1], -1, -1, tensor["shape"][4]]
    result.append([x, filter, bias, y, ksize, strides, padding, data_format, offset_x, kernel_name])
    return result


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=invalid-name,redefined-builtin,too-many-locals,unused-argument,unused-variable,unnecessary-lambda
def check_supported(x, filter, bias, y, ksize, strides,
                    padding="VALID", data_format="NHWC", offset_x=0,
                    kernel_name="avg_pool"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    True or False
    """
    ori_shape = y.get("ori_shape")
    if data_format == "NHWC":
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        outputh = ori_shape[1]
        outputw = ori_shape[2]
    elif data_format == "NCHW":
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        outputh = ori_shape[2]
        outputw = ori_shape[3]
    else:
        return False
    is_support_kernel = (ksize_h * ksize_w <= AVG_KERNEL_SIZE_H_MUL_W) or \
                        (ksize_h <= AVG_KERNEL_SIZE and ksize_w <= AVG_KERNEL_SIZE)
    if not is_support_kernel and outputh != 1 and outputw == 1:
        return False
    if not is_support_kernel and not (outputh == 1 and outputw == 1):
        return False
    return True


def get_op_support_info(x, filter, bias, y, ksize, strides,
                        padding="VALID", data_format="NHWC", offset_x=0,
                        kernel_name="avg_pool"):
    """
    get the avgpool split
    """
    format_x = x.get("format")
    input_shape = x.get("shape")

    if data_format in ("NHWC",):
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        window = [ksize[1], ksize[2]]
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        window = [ksize[2], ksize[3]]

    if format_x == "NC1HWC0":
        if (ksize_h == window[0] and ksize_w == window[1]) or padding == "SAME":
            axis_split_matrix = [[util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
                                 util_select_op_base.SplitOutput([0, [0]])]]
        elif padding == "VALID":
            axis_split_matrix = [
                [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                [util_select_op_base.SplitInput([0, [2], [0], [0]]), util_select_op_base.SplitOutput([0, [2]])]]
        else:
            axis_split_matrix = None
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list, 2, 0)

    return op_cal_info_in_json


# pylint: disable=locally-disabled,too-many-arguments,too-many-statements
# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _check_window_rule(ksize, strides, padding, data_format, offset_x):
    """
    check ksize and strides of window in pooling
    :param ksize: list or tuple, the length must be 4
    :param strides: list or tuple, the length must be 4
    :param data_format: input format
    :return: None
    """
    if len(ksize) != 4:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'ksize'
        error_info['min_value'] = '4'
        error_info['max_value'] = '4'
        error_info['real_value'] = len(ksize)
        raise RuntimeError(error_info,
                           "In op[%s], the num of dimensions of input[%s] "
                           "should be in the range of [%s, %s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['real_value']))

    if len(strides) != 4:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'strides'
        error_info['min_value'] = '4'
        error_info['max_value'] = '4'
        error_info['real_value'] = len(strides)
        raise RuntimeError(error_info,
                           "In op[%s], the num of dimensions of input[%s] "
                           "should be in the range of [%s, %s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['real_value']))

    ksize_c = ksize[3] if data_format in ("NHWC",) else ksize[1]
    strides_c = strides[3] if data_format in ("NHWC",) else strides[1]
    if ksize[0] != 1 or (ksize_c != 1):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = ",".join(("ksize[1]", "ksize[3]"))
        error_info['expected_value'] = '1'
        error_info['real_value'] = ",".join((str(ksize[1]), str(ksize[3])))
        raise RuntimeError("In op[%s], the parameter[%s] should be [%s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['expected_value'],
                            error_info['real_value']))


    if strides[0] != 1 or strides_c != 1:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = ",".join(("strides[1]", "strodes[3]"))
        error_info['expected_value'] = '1'
        error_info['real_value'] = ",".join((str(strides[1]), str(strides[3])))
        raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be [%s], "
                           "but actually is [%s]." % (error_info['op_name'],
                                                      error_info['param_name'],
                                                      error_info['expected_value'],
                                                      error_info['real_value']))

    if padding not in ("SAME", "VALID"):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_015
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'padding'
        error_info['expected_value_list'] = ",".join(("SAME", "VALID"))
        error_info['real_value'] = padding
        raise RuntimeError(error_info, "In op[%s], parameter[%s] should be one of [%s], "
                            "but actually is [%s]." % (error_info['op_name'],
                                                       error_info['param_name'],
                                                       error_info['expected_value_list'],
                                                       error_info['real_value']))

    if data_format not in("NCHW", "NHWC", "NC1HWC0"):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_015
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'x'
        error_info['excepted_format_list'] = ",".join(("NC1HWC0",
                                                       "NCHW", "NHWC"))
        error_info['format'] = data_format
        raise RuntimeError(error_info, "In op[%s], the format[%s] of input "
                                       "should be one of [%s], "
                                       "but actuall"
                                       "y is [%s]."
                           % (error_info['op_name'],
                              error_info['param_name'],
                              error_info['excepted_format_list'],
                              error_info['format']))

    if offset_x != 0:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'offset_x'
        error_info['expected_value'] = '0'
        error_info['real_value'] = str(offset_x)
        raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be [%s], "
                                       "but actually is [%s]."
                           % (error_info['op_name'],
                              error_info['param_name'],
                              error_info['expected_value'],
                              error_info['real_value']))


def _avg_pool_check_rule(input_shape, input_dtype, output_dtype,
                          ksize, strides, padding, data_format, offset_x, kernel_name):
    """
    :param input_shape: shape of input_data
    :param input_dtype: dtype of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of avgpooling
    :param strides: the stride of avgpooling window
    :param padding: padding_mode of avgpool
    :param data_format: NHWC default
    :param offset_x: default 0
    :param kernel_name: cce kernel name
    :return: None
    """
    if len(input_shape) != INPUT_DIM:
        error_manager_cube.raise_err_specific_user("avg_pool",
                                                   "ori_input_shape dim should be 4.")
    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ["float16", "int8"])
    para_check.check_dtype(output_dtype, ["float16", "int8", "int32"])

    _check_window_rule(ksize, strides, padding, data_format, offset_x)


def _check_filter_window(fmap, filter, window, stride):
    fmap_shape = fmap.get("ori_shape")
    filter_shape = filter.get("ori_shape")
    filter_format = filter.get("ori_format")
    if stride[0] > MAX_CUBE_STRIDE or stride[1] > MAX_CUBE_STRIDE:
        raise RuntimeError("In op[%s], the [%s] should less than [%s] when filter is None"
                           % ('avgpool', 'stride', str(MAX_CUBE_STRIDE)))
    if filter_format not in ("NCHW", "NHWC"):
        raise RuntimeError("In op[%s], the ori_format of filter "
                                       "should be [%s] or [%s]"
                           % ('avgpool', 'NCHW', 'NHWC'))
    h_index = filter_format.index("H")
    w_index = filter_format.index("W")
    c_index = filter_format.index("C")
    n_index = filter_format.index("N")
    if filter_shape[h_index] != window[0] or filter_shape[w_index] != window[1]:
        raise RuntimeError("In op[%s], the h_shape of filter "
                                       "should be equal with [%s],"
                           % ('avgpool', 'ksize'))
    if filter_shape[c_index] != 1:
        raise RuntimeError("In op[%s], the c_shape of filter "
                                       "should be [%s],"
                           % ('avgpool', '1'))
    if filter_shape[n_index] != fmap_shape[fmap.get("ori_format").index("C")]:
        raise RuntimeError("In op[%s], the N shape of filter "
                                       "should be equal with C shape of fmap,"
                           % ('avgpool'))


@register_operator("AvgPool")
@para_check.check_input_type(dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list),
                             str, str, int, str)
def avg_pool(x, filter, bias, y, ksize, strides,
             padding="VALID", data_format="NHWC", offset_x=0,
             kernel_name="avg_pool"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    None
    """

    # get shape&dtype
    # input_shape only support format NCHW
    input_shape = x.get("ori_shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    input_format = x.get("ori_format")
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    output_format = y.get("ori_format")
    if not output_format:
        y["ori_format"] = input_format
    elif output_format not in ("NCHW", "NHWC"):
        error_manager_cube.raise_err_one_para("E62006", "avg_pool",
                                              "output_format should be 'NCHW or 'NHWC'")

    _avg_pool_check_rule(input_shape, input_dtype, output_dtype, ksize, strides, padding,
                         data_format, offset_x, kernel_name)
    if input_format == "NCHW":
        input_c, input_h, input_w = input_shape[1:4]
        stride = [-1, -1, strides[2], strides[3]]
        window = [ksize[2], ksize[3]]
        strides_h = strides[2]
        strides_w = strides[3]

    elif input_format == "NHWC":
        input_h, input_w, input_c = input_shape[1:4]
        stride = [-1, strides[1], strides[2], -1]
        window = [ksize[1], ksize[2]]
        strides_h = strides[1]
        strides_w = strides[2]
    else:
        raise RuntimeError("Unsupported input format!")
    tbe_context.get_context().add_compile_info("strides_h", strides_h)
    tbe_context.get_context().add_compile_info("strides_w", strides_w)

    if bias is None and filter is not None:
        dilations = (1, 1, 1, 1)
        _check_filter_window(x, filter, window, stride)
        offset_w = None
        pad = padding
        conv2d(x, filter, bias, offset_w, y, stride, pad, dilations,
               groups=input_c, data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)
    else:
        if filter is None:
            obj = AvgPool(input_dtype, window, [strides_h, strides_w], padding, kernel_name)
            obj.avg_pool_operator()
        if bias is not None:
            error_manager_cube.raise_err_input_params_not_expected("dynamic_avg_pool", "bias", "None",
                                                                   "dict")
