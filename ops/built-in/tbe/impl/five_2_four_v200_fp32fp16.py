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

five_2_four_v200_32b
"""
# ’pylint: disable=too-many-lines
from te import tik


# ’pylint: disable=too-many-instance-attributes,too-few-public-methods
class VnchwConv(object):
    """
    VnchwConv
    """

    # ’pylint: disable=no-self-use,
    @staticmethod
    def _align16(value):
        """
        Calculate align 16 size of value
        Parameters
        ----------
        value : int
            input int to align 16
        """

        return (value + 15) // 16 * 16

    # ’pylint: disable=invalid-name
    def __init__(self, src_dict, dst_dict, kernel_name_value):
        """
        init class VnchwConv
        src_size,
        dst_size,
        input gm,
        output gm
        """

        self.ai_core_arch = "v200"
        self.ai_core_version = "aic"
        self.tik_dprofile = tik.Dprofile(self.ai_core_arch, self.ai_core_version)
        self.tik_instance = tik.Tik(self.tik_dprofile)
        self.UBSIZE = self.tik_dprofile.get_unified_buffer_size()
        self.C0_const = 16
        self.src_shape = src_dict.get("shape")
        self.dst_shape = dst_dict.get("shape")
        self.dtype_size = 2
        self.dtype = src_dict.get("dtype")
        if self.dtype == "float32":
            self.dtype_size = 4
        src_size = [
            self.src_shape[0],
            self.src_shape[1],
            self.src_shape[2],
            self.src_shape[3],
            self.src_shape[4]]

        dst_size = [
            self.dst_shape[0],
            self.dst_shape[1],
            self.dst_shape[2],
            self.dst_shape[3]]

        self.kernel_name = kernel_name_value
        self.src_batch = src_size[0]
        self.src_channel1 = src_size[1]
        self.src_height = src_size[2]
        self.src_width = src_size[3]
        self.src_channel0 = src_size[4]

        self.dst_batch = dst_size[0]
        self.dst_channel = dst_size[1]
        self.dst_height = dst_size[2]
        self.dst_width = dst_size[3]

        self.aicore_use = 8

        if self.src_height < 0 \
                or self.src_width < 0 \
                or self.dst_height < 0 \
                or self.dst_width < 0:
            raise RuntimeError("src_size or dst_size is illegal")
        if src_dict.get("dtype") != dst_dict.get("dtype"):
            raise RuntimeError("src dtype != dst dtype")

        self.src_image = self.tik_instance.Tensor(
            self.dtype,
            (self.src_batch,
             self.src_channel1,
             self.src_height,
             self.src_width,
             self.src_channel0),
            name="src_image",
            scope=tik.scope_gm)

        self.dst_image = self.tik_instance.Tensor(
            self.dtype,
            (self.dst_batch,
             self.dst_channel,
             self.dst_height,
             self.dst_width),
            name="dst_image",
            scope=tik.scope_gm)

        self.chw = self.dst_channel * self.dst_height * self.dst_width
        self.hw = self.dst_height * self.dst_width
        self.w = self.dst_width

    # ’pylint: disable=too-many-nested-blocks,too-many-branches,too-many-statements
    def compute_slice(self):
        """
        Calculate the split parameters according to different shapes
        for ub buf size
        """
        size_c0 = self.dtype_size * self.src_channel0
        size_hwc0 = self.dtype_size \
                    * self._align16(self.src_height * self.src_width) \
                    * self.src_channel0
        size_c1hwc0 = self.dtype_size * self.src_channel1 \
                      * self._align16(self.src_height * self.src_width) \
                      * self.src_channel0

        ub_image_size = (self.UBSIZE - 1024) // 2

        if size_c1hwc0 <= ub_image_size:
            if self.dtype == "float16":
                self._compute_core_all_fp16()
            elif self.dtype == "float32":
                self._compute_core_all_fp32()
        else:
            if size_hwc0 <= ub_image_size:
                c1_one_slice = ub_image_size // size_hwc0
                c1_repeat = self.src_channel1 // c1_one_slice
                c1_tail = self.src_channel1 % c1_one_slice
                dst_channel = c1_one_slice * self.C0_const

                if c1_tail == 0:
                    if self.dst_channel % \
                            (c1_one_slice * c1_repeat * self.C0_const) == 0:
                        for c1_repeat_idx in range(c1_repeat):
                            if self.dtype == "float16":
                                self._compute_core_c1_fp16(
                                    c1_repeat_idx * c1_one_slice,
                                    c1_one_slice,
                                    dst_channel)

                            elif self.dtype == "float32":
                                self._compute_core_c1_fp32(
                                    c1_repeat_idx * c1_one_slice,
                                    c1_one_slice,
                                    dst_channel)

                    else:
                        for c1_repeat_idx in range(c1_repeat - 1):
                            if self.dtype == "float16":
                                self._compute_core_c1_fp16(
                                    c1_repeat_idx * c1_one_slice,
                                    c1_one_slice,
                                    dst_channel)

                            elif self.dtype == "float32":
                                self._compute_core_c1_fp32(
                                    c1_repeat_idx * c1_one_slice,
                                    c1_one_slice,
                                    dst_channel)

                        dst_channel_tail = self.dst_channel \
                                           - c1_one_slice * (c1_repeat - 1) * self.C0_const
                        if self.dtype == "float16":
                            self._compute_core_c1_fp16(
                                (c1_repeat - 1) * c1_one_slice,
                                c1_one_slice,
                                dst_channel_tail)

                        elif self.dtype == "float32":
                            self._compute_core_c1_fp32(
                                (c1_repeat - 1) * c1_one_slice,
                                c1_one_slice,
                                dst_channel_tail)

                elif c1_tail > 0:
                    for c1_repeat_idx in range(c1_repeat):
                        if self.dtype == "float16":
                            self._compute_core_c1_fp16(
                                c1_repeat_idx * c1_one_slice,
                                c1_one_slice,
                                dst_channel)

                        elif self.dtype == "float32":
                            self._compute_core_c1_fp32(
                                c1_repeat_idx * c1_one_slice,
                                c1_one_slice,
                                dst_channel)

                    dst_channel_tail = self.dst_channel \
                                       - c1_one_slice * c1_repeat * self.C0_const
                    if self.dtype == "float16":
                        self._compute_core_c1_fp16(
                            c1_repeat * c1_one_slice,
                            c1_tail,
                            dst_channel_tail)

                    elif self.dtype == "float32":
                        self._compute_core_c1_fp32(
                            c1_repeat * c1_one_slice,
                            c1_tail,
                            dst_channel_tail)

            else:
                hw_one_slice = ub_image_size // size_c0 // self.C0_const * self.C0_const
                hw_repeat = (self.src_height * self.src_width + hw_one_slice - 1) \
                            // hw_one_slice
                hw_tail = (self.src_height * self.src_width) % hw_one_slice
                if hw_tail == 0:
                    for hw_repeat_idx in range(hw_repeat):
                        if self.dtype == "float16":
                            self._compute_core_hw_fp16(
                                hw_repeat_idx * hw_one_slice,
                                hw_one_slice)

                        elif self.dtype == "float32":
                            self._compute_core_hw_fp32(
                                hw_repeat_idx * hw_one_slice,
                                hw_one_slice)

                elif hw_tail > 0:
                    if self.dtype == "float16":
                        self._compute_core_hw_fp16(
                            (hw_repeat - 1) * hw_one_slice,
                            hw_tail)

                    elif self.dtype == "float32":
                        self._compute_core_hw_fp32(
                            (hw_repeat - 1) * hw_one_slice,
                            hw_tail)

                    for hw_repeat_idx in range(hw_repeat - 1):
                        if self.dtype == "float16":
                            self._compute_core_hw_fp16(
                                hw_repeat_idx * hw_one_slice,
                                hw_one_slice)

                        elif self.dtype == "float32":
                            self._compute_core_hw_fp32(
                                hw_repeat_idx * hw_one_slice,
                                hw_one_slice)

    def _compute_core_hw_fp16(self, hw_st, hw_tail):
        """
        Calculate five 2 four result cut input shape at hw dimension
        for dtype fp16

        Parameters
        ----------
        hw_st : int
            hw start point at hw dimension
        hw_tail : int
            length of one part from start point to the end
        """
        with self.tik_instance.new_stmt_scope():
            input_ub_fp16 = self.tik_instance.Tensor(
                self.dtype,
                (self._align16(hw_tail),
                 self.src_channel0),
                name="input_ub_fp16",
                scope=tik.scope_ubuf)

            output_ub_fp16 = self.tik_instance.Tensor(
                self.dtype,
                (self.src_channel0,
                 self._align16(hw_tail)),
                name="output_ub_fp16",
                scope=tik.scope_ubuf)

            src_hw_nbrust = hw_tail
            h_st = hw_st // self.src_width
            w_st = hw_st % self.src_width

            with self.tik_instance.for_range(0, self.src_batch) as index_n:
                with self.tik_instance.for_range(0, self.src_channel1) \
                        as index_c:
                    self.tik_instance.data_move(
                        input_ub_fp16[0, 0],
                        self.src_image[index_n, index_c, h_st, w_st, 0],
                        0,
                        1,
                        src_hw_nbrust,
                        0,
                        0)

                    dist_list = [output_ub_fp16[i, 0] for i in range(0, 16)]
                    src_list = [input_ub_fp16[i, 0] for i in range(0, 16)]

                    if self._align16(hw_tail) // self.C0_const > 1:
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            self._align16(hw_tail) // self.C0_const,
                            1,
                            16)
                    else:
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            1,
                            0,
                            0)
                    dst_channel_one_loop = self.tik_instance.Scalar("int32")
                    dst_channel_one_loop.set_as(self.C0_const)
                    if self.dst_channel % self.C0_const > 0:
                        # have c1 tail
                        with self.tik_instance.if_scope(
                                index_c == self.src_channel1 - 1):
                            dst_channel_one_loop.set_as(
                                self.dst_channel % self.C0_const)
                    with self.tik_instance.for_range(0, dst_channel_one_loop) \
                            as dst_index_c:
                        self.tik_instance.data_move(
                            self.dst_image[index_n * self.chw +
                                           (index_c * self.C0_const + dst_index_c) * self.hw +
                                           h_st * self.w +
                                           w_st],
                            output_ub_fp16[dst_index_c, 0],
                            0,
                            self._align16(hw_tail) // self.C0_const,
                            1,
                            0,
                            0)

    # ’pylint: disable=too-many-locals,invalid-name,too-many-statements
    def _compute_core_hw_fp32(self, hw_st, hw_tail):
        """
        Calculate five 2 four result cut input shape at hw dimension
        for dtype fp32

        Parameters
        ----------
        hw_st : int
            hw start point at hw dimension
        hw_tail : int
            length of one part from start point to the end
        """

        with self.tik_instance.new_stmt_scope():
            input_ub_fp32 = self.tik_instance.Tensor(
                self.dtype,
                (self._align16(hw_tail),
                 self.src_channel0),
                name="input_ub_fp32",
                scope=tik.scope_ubuf)

            output_ub_fp32 = self.tik_instance.Tensor(
                self.dtype,
                (self.src_channel0,
                 self._align16(hw_tail)),
                name="output_ub_fp32",
                scope=tik.scope_ubuf)

            src_hw_nbrust = hw_tail * 2
            dst_hw_nbrust = self._align16(hw_tail) // 16
            h_st = hw_st // self.src_width
            w_st = hw_st % self.src_width
            with self.tik_instance.for_range(0, self.src_batch) \
                    as index_n:
                with self.tik_instance.for_range(0, self.src_channel1 - 1) \
                        as index_c:
                    self.tik_instance.data_move(
                        input_ub_fp32[0, 0],
                        self.src_image[index_n,
                                       index_c,
                                       h_st,
                                       w_st,
                                       0],
                        0,
                        1,
                        src_hw_nbrust,
                        0,
                        0)
                    with self.tik_instance.for_range(0, dst_hw_nbrust) \
                            as index_hwm:
                        src_list = [input_ub_fp32[i + index_hwm * 16, 0]
                                    for i in range(0, 16)]
                        dist_list = [output_ub_fp32[i, index_hwm * 16]
                                     for i in range(0, 16)]
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            2,
                            1,
                            1)

                    dst_c_st = index_c * 16
                    dst_c_end = (index_c + 1) * 16
                    with self.tik_instance.for_range(dst_c_st, dst_c_end) \
                            as index_cm:
                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = index_cm * self.dst_height * self.dst_width \
                                    + h_st * self.dst_width + w_st
                        dst_channel_one_loop = self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)
                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 * 8 >= \
                                total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(dst_hw_nbrust - num_sub)
                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    index_n * self.chw +
                                    index_cm * self.hw +
                                    h_st * self.w +
                                    w_st],
                                output_ub_fp32[(index_cm - dst_c_st) % 8 \
                                               * 2,
                                               (index_cm - dst_c_st) // 8 * 8],
                                0,
                                dst_hw_nbrust,
                                1,
                                1,
                                1)

                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = index_cm * self.dst_height * self.dst_width \
                                    + (h_st + (w_st + 8) // self.dst_width) \
                                    * self.dst_width + (w_st + 8) % self.dst_width

                        dst_channel_one_loop = self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 * 8 >= \
                                total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(dst_hw_nbrust - num_sub)

                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    index_n * self.chw +
                                    index_cm * self.hw +
                                    (h_st + ((w_st + 8) // self.dst_width)) * self.w +
                                    (w_st + 8) % self.dst_width],
                                output_ub_fp32[(index_cm - dst_c_st) % 8 \
                                               * 2 + 1,
                                               (index_cm - dst_c_st) // 8 * 8],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

                with self.tik_instance.for_range(
                        self.src_channel1 - 1, self.src_channel1) as index_c:
                    self.tik_instance.data_move(
                        input_ub_fp32[
                            0,
                            0],
                        self.src_image[
                            index_n,
                            index_c,
                            h_st,
                            w_st,
                            0],
                        0,
                        1,
                        src_hw_nbrust,
                        0,
                        0)
                    with self.tik_instance.for_range(0, dst_hw_nbrust) \
                            as index_hwm:
                        src_list = [input_ub_fp32[i + index_hwm * 16, 0]
                                    for i in range(0, 16)]
                        dist_list = [output_ub_fp32[i, index_hwm * 16]
                                     for i in range(0, 16)]
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            2,
                            1,
                            1)

                    dst_c_st = index_c * 16
                    dst_c_end = self.dst_channel

                    with self.tik_instance.for_range(dst_c_st, dst_c_end) \
                            as index_cm:
                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = index_cm * self.dst_height * self.dst_width \
                                    + h_st * self.dst_width + w_st
                        dst_channel_one_loop = self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)
                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 * 8 >= \
                                total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(dst_hw_nbrust - num_sub)
                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[index_n * self.chw +
                                               index_cm * self.hw +
                                               h_st * self.w +
                                               w_st],
                                output_ub_fp32[
                                    (index_cm - dst_c_st) % 8 * 2,
                                    (index_cm - dst_c_st) // 8 * 8],
                                0,
                                dst_hw_nbrust,
                                1,
                                1,
                                1)

                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = index_cm * self.dst_height * self.dst_width \
                                    + (h_st + (w_st + 8) // self.dst_width) \
                                    * self.dst_width + (w_st + 8) % self.dst_width
                        dst_channel_one_loop = self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 * 8 >= \
                                total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 15) // 16
                            dst_channel_one_loop.set_as(dst_hw_nbrust - num_sub)
                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    index_n * self.chw +
                                    index_cm * self.hw +
                                    (h_st + ((w_st + 8) // self.dst_width)) * self.w +
                                    (w_st + 8) % self.dst_width],
                                output_ub_fp32
                                [(index_cm - dst_c_st) % 8 * 2 + 1,
                                 (index_cm - dst_c_st) // 8 * 8],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

    # ’pylint: disable=too-many-locals,invalid-name,too-many-statements
    def _compute_core_c1_fp16(self, c1_st, c1_one_slice, dst_channel):
        """
        Calculate five 2 four result cut input shape at c1 dimension
        for dtype fp16

        Parameters
        ----------
        c1_st : int
            c1 start point at c1 dimension
        c1_one_slice : int
            length of one part from start point to the end
        dst_channel: int
            the end of the dst channel for this part
        """

        with self.tik_instance.new_stmt_scope():
            input_ub_fp16 = self.tik_instance.Tensor(
                self.dtype,
                (c1_one_slice,
                 self._align16(self.src_height * self.src_width),
                 self.src_channel0),
                name="input_ub_fp16",
                scope=tik.scope_ubuf)

            output_ub_fp16 = self.tik_instance.Tensor(
                self.dtype,
                (c1_one_slice * self.src_channel0,
                 self._align16(self.dst_height * self.dst_width)),
                name="output_ub_fp16",
                scope=tik.scope_ubuf)

            dst_len = self._align16(dst_channel)
            dst_c_nbrust = dst_len // self.C0_const
            src_hw_nbrust = self._align16(
                self.src_height * self.src_width * self.src_channel0) \
                            // self.C0_const
            dst_hw_nbrust = self._align16(self.dst_height * self.dst_width) \
                            // self.C0_const

            with self.tik_instance.for_range(0, self.src_batch) as index_n:
                with self.tik_instance.for_range(0, c1_one_slice) as index_c:
                    self.tik_instance.data_move(
                        input_ub_fp16[
                            index_c,
                            0,
                            0],
                        self.src_image[
                            index_n,
                            c1_st + index_c,
                            0,
                            0,
                            0],
                        0,
                        1,
                        src_hw_nbrust,
                        0,
                        0)

                with self.tik_instance.for_range(0, dst_c_nbrust) as index_c:
                    dist_list = [output_ub_fp16[i + index_c * 16, 0]
                                 for i in range(0, 16)]
                    src_list = [input_ub_fp16[index_c, i, 0]
                                for i in range(0, 16)]
                    if dst_hw_nbrust < 2:
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            1,
                            0,
                            0)

                    elif dst_hw_nbrust > 1:
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            dst_hw_nbrust,
                            1,
                            16)

                with self.tik_instance.for_range(0, dst_channel) as index_c:
                    self.tik_instance.data_move(
                        self.dst_image[
                            index_n * self.chw +
                            (c1_st * self.C0_const + index_c) * self.hw +
                            0 +
                            0],
                        output_ub_fp16[
                            index_c,
                            0],
                        0,
                        1,
                        dst_hw_nbrust,
                        0,
                        0)

    # ’pylint: disable=too-many-locals,invalid-name,too-many-statements
    def _compute_core_c1_fp32(self, c1_st, c1_one_slice, dst_channel):
        """
        Calculate five 2 four result cut input shape at c1 dimension
        for dtype fp32

        Parameters
        ----------
        c1_st : int
            c1 start point at c1 dimension
        c1_one_slice : int
            length of one part from start point to the end
        dst_channel: int
            the end of the dst channel for this part
        """

        with self.tik_instance.new_stmt_scope():
            input_ub_fp32 = self.tik_instance.Tensor(
                self.dtype,
                (c1_one_slice,
                 self._align16(self.src_height * self.src_width),
                 self.src_channel0),
                name="input_ub_fp32",
                scope=tik.scope_ubuf)

            output_ub_fp32 = self.tik_instance.Tensor(
                self.dtype,
                (c1_one_slice * self.src_channel0,
                 self._align16(self.dst_height * self.dst_width)),
                name="output_ub_fp32",
                scope=tik.scope_ubuf)

            temp = self.tik_instance.Tensor(
                self.dtype,
                (32,),
                name="temp",
                scope=tik.scope_ubuf)

            dst_len = self._align16(dst_channel)
            dst_c_nbrust = dst_len // self.C0_const
            src_hw_nbrust = self._align16(self.src_height * self.src_width \
                                          * self.src_channel0) // 8
            dst_hw_nbrust = self._align16(self.dst_height * self.dst_width) \
                            // self.C0_const

            with self.tik_instance.for_range(0, self.src_batch) as index_n:
                with self.tik_instance.if_scope(index_n < self.src_batch - 1):
                    self.tik_instance.data_move(
                        temp,
                        self.dst_image[
                            (index_n + 1) * self.chw +
                            0 +
                            0 +
                            0],
                        0,
                        1,
                        1,
                        0,
                        0)

                with self.tik_instance.for_range(0, c1_one_slice) \
                        as index_c:
                    self.tik_instance.data_move(
                        input_ub_fp32[
                            index_c,
                            0,
                            0],
                        self.src_image[
                            index_n,
                            c1_st + index_c,
                            0,
                            0,
                            0],
                        0,
                        1,
                        src_hw_nbrust,
                        0,
                        0)

                with self.tik_instance.for_range(0, dst_c_nbrust - 1) as index_c:
                    with self.tik_instance.for_range(0, dst_hw_nbrust) as index_hwm:
                        src_list = [input_ub_fp32[index_c, i + index_hwm * 16, 0] for i in range(0, 16)]
                        dist_list = [output_ub_fp32[i + index_c * 16, index_hwm * 16] for i in range(0, 16)]
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            2,
                            1,
                            1)

                    dst_c_st = (index_c + c1_st) * 16
                    dst_c_end = (index_c + 1 + c1_st) * 16

                    with self.tik_instance.for_range(dst_c_st, dst_c_end) \
                            as index_cm:
                        dst_n = index_n
                        dst_c = index_cm
                        dst_h = 0
                        dst_w = 0
                        src_c = (index_cm - dst_c_st) % 8 * 2 + index_c * 16
                        src_hw = (index_cm - dst_c_st) // 8 * 8

                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = dst_c * self.dst_height * self.dst_width \
                                    + dst_h * self.dst_width + dst_w
                        dst_channel_one_loop = \
                            self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 * 8 >= \
                                total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(
                                dst_hw_nbrust - num_sub)

                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    dst_n * self.chw +
                                    dst_c * self.hw +
                                    dst_h * self.w +
                                    dst_w],
                                output_ub_fp32[
                                    src_c,
                                    src_hw],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

                        dst_n = index_n
                        dst_c = index_cm
                        dst_h = 8 // self.dst_width
                        dst_w = 8 % self.dst_width
                        src_c = (index_cm - dst_c_st) % 8 * 2 + 1 + index_c * 16
                        src_hw = (index_cm - dst_c_st) // 8 * 8

                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = dst_c * self.dst_height * self.dst_width \
                                    + dst_h * self.dst_width + dst_w
                        dst_channel_one_loop = \
                            self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 * 8 >=
                                total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(
                                dst_hw_nbrust - num_sub)

                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    dst_n * self.chw +
                                    dst_c * self.hw +
                                    dst_h * self.w +
                                    dst_w],
                                output_ub_fp32[
                                    src_c,
                                    src_hw],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

                with self.tik_instance.for_range(dst_c_nbrust - 1,
                                                 dst_c_nbrust) \
                        as index_c:
                    with self.tik_instance.for_range(0, dst_hw_nbrust) \
                            as index_hwm:
                        src_list = [input_ub_fp32[index_c, i + index_hwm * 16, 0]
                                    for i in range(0, 16)]
                        dist_list = [output_ub_fp32[i + index_c * 16, index_hwm * 16]
                                     for i in range(0, 16)]
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            2,
                            1,
                            1)

                    dst_c_st = (index_c + c1_st) * 16
                    dst_c_end = c1_st * 16 + dst_channel

                    with self.tik_instance.for_range(dst_c_st, dst_c_end) \
                            as index_cm:
                        dst_n = index_n
                        dst_c = index_cm
                        dst_h = 0
                        dst_w = 0
                        src_c = (index_cm - dst_c_st) % 8 * 2 + index_c * 16
                        src_hw = (index_cm - dst_c_st) // 8 * 8

                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = dst_c * self.dst_height * self.dst_width \
                                    + dst_h * self.dst_width + dst_w
                        dst_channel_one_loop = \
                            self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 * 8 >=
                                total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(
                                dst_hw_nbrust - num_sub)

                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    dst_n * self.chw +
                                    dst_c * self.hw +
                                    dst_h * self.w +
                                    dst_w],
                                output_ub_fp32[
                                    src_c,
                                    src_hw],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

                        dst_n = index_n
                        dst_c = index_cm
                        dst_h = 8 // self.dst_width
                        dst_w = 8 % self.dst_width
                        src_c = (index_cm - dst_c_st) % 8 \
                                * 2 + 1 + index_c * 16
                        src_hw = (index_cm - dst_c_st) // 8 * 8
                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = dst_c * self.dst_height * self.dst_width \
                                    + dst_h * self.dst_width + dst_w
                        dst_channel_one_loop = \
                            self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)
                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 * 8 >=
                                total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(dst_hw_nbrust - num_sub)
                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    dst_n * self.chw +
                                    dst_c * self.hw +
                                    dst_h * self.w +
                                    dst_w],
                                output_ub_fp32[
                                    src_c,
                                    src_hw],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

                with self.tik_instance.if_scope(index_n < self.src_batch - 1):
                    self.tik_instance.data_move(
                        self.dst_image[
                            (index_n + 1) * self.chw +
                            0 +
                            0 +
                            0],
                        temp,
                        0,
                        1,
                        1,
                        0,
                        0)

    # ’pylint: disable=too-many-locals,too-many-statements
    # ’pylint: disable=invalid-name
    def _compute_core_all_fp16(self):
        """
        Calculate five 2 four result without cutting input shape
        for dtype fp16
        """

        with self.tik_instance.new_stmt_scope():
            input_ub_fp16 = self.tik_instance.Tensor(self.dtype,
                                                     (self.src_channel1,
                                                      self._align16(self.src_height * self.src_width),
                                                      self.src_channel0),
                                                     name="input_ub_fp16",
                                                     scope=tik.scope_ubuf)

            output_ub_fp16 = self.tik_instance.Tensor(self.dtype,
                                                      (self._align16(self.dst_channel),
                                                       self._align16(self.dst_height * self.dst_width)),
                                                      name="output_ub_fp16",
                                                      scope=tik.scope_ubuf)

            dst_len = self._align16(self.dst_channel)
            dst_c_nbrust = dst_len // 16
            src_hw_nbrust = self._align16(
                self.src_height * self.src_width * self.src_channel0) // 16
            dst_hw_nbrust = self._align16(
                self.dst_height * self.dst_width) // 16

            with self.tik_instance.for_range(0, self.src_batch) as index_n:
                with self.tik_instance.for_range(0, self.src_channel1) \
                        as index_c:
                    self.tik_instance.data_move(
                        input_ub_fp16[
                            index_c,
                            0,
                            0],
                        self.src_image[
                            index_n,
                            index_c,
                            0,
                            0,
                            0],
                        0,
                        1,
                        src_hw_nbrust,
                        0,
                        0)

                with self.tik_instance.for_range(0, dst_c_nbrust) as index_c:
                    dist_list = [output_ub_fp16[i + index_c * 16, 0]
                                 for i in range(0, 16)]
                    src_list = [input_ub_fp16[index_c, i, 0]
                                for i in range(0, 16)]
                    if dst_hw_nbrust < 2:
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            1,
                            0,
                            0)

                    elif dst_hw_nbrust > 1:
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            dst_hw_nbrust,
                            1,
                            16)

                with self.tik_instance.for_range(0, self.dst_channel) \
                        as index_c:
                    self.tik_instance.data_move(
                        self.dst_image[
                            index_n * self.chw +
                            index_c * self.hw +
                            0 +
                            0],
                        output_ub_fp16[
                            index_c,
                            0],
                        0,
                        1,
                        dst_hw_nbrust,
                        0,
                        0)

    # ’pylint: disable=too-many-locals,invalid-name,too-many-statements
    def _compute_core_all_fp32(self):
        """
        Calculate five 2 four result without cutting input shape
        for dtype fp32

        """

        with self.tik_instance.new_stmt_scope():
            input_ub_fp32 = self.tik_instance.Tensor(self.dtype,
                                                     (self.src_channel1,
                                                      self._align16(self.src_height * self.src_width),
                                                      self.src_channel0),
                                                     name="input_ub_fp32",
                                                     scope=tik.scope_ubuf)

            output_ub_fp32 = self.tik_instance.Tensor(self.dtype,
                                                      (self._align16(self.dst_channel),
                                                       self._align16(self.dst_height * self.dst_width)),
                                                      name="output_ub_fp32",
                                                      scope=tik.scope_ubuf)

            dst_len = self._align16(self.dst_channel)
            dst_c_nbrust = dst_len // 16
            src_hw_nbrust = self._align16(self.src_height * self.src_width
                                          * self.src_channel0) // 8
            dst_hw_nbrust = self._align16(
                self.dst_height * self.dst_width) // 16

            with self.tik_instance.for_range(0, self.src_batch) \
                    as index_n:
                with self.tik_instance.for_range(0, self.src_channel1) \
                        as index_c:
                    self.tik_instance.data_move(
                        input_ub_fp32[
                            index_c,
                            0,
                            0],
                        self.src_image[
                            index_n,
                            index_c,
                            0,
                            0,
                            0],
                        0,
                        1,
                        src_hw_nbrust,
                        0,
                        0)

                with self.tik_instance.for_range(0, dst_c_nbrust - 1) \
                        as index_c:
                    with self.tik_instance.for_range(0, dst_hw_nbrust) \
                            as index_hwm:
                        src_list = [input_ub_fp32[index_c, i + index_hwm * 16, 0]
                                    for i in range(0, 16)]
                        dist_list = [output_ub_fp32[i + index_c * 16, index_hwm * 16]
                                     for i in range(0, 16)]
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            2,
                            1,
                            1)

                    dst_c_st = index_c * 16
                    dst_c_end = (index_c + 1) * 16

                    with self.tik_instance.for_range(dst_c_st, dst_c_end) \
                            as index_cm:
                        dst_n = index_n
                        dst_c = index_cm
                        dst_h = 0
                        dst_w = 0
                        src_c = (index_cm - dst_c_st) % 8 * 2 + dst_c_st
                        src_hw = (index_cm - dst_c_st) // 8 * 8
                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = dst_c * self.dst_height * self.dst_width \
                                    + dst_h * self.dst_width + dst_w
                        dst_channel_one_loop = \
                            self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(move_size \
                                                        + (dst_hw_nbrust - 1) * 2 \
                                                        * 8 >= total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(
                                dst_hw_nbrust - num_sub)
                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    dst_n * self.chw +
                                    dst_c * self.hw +
                                    dst_h * self.w +
                                    dst_w],
                                output_ub_fp32[
                                    src_c,
                                    src_hw],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

                        dst_n = index_n
                        dst_c = index_cm
                        dst_h = 8 // self.dst_width
                        dst_w = 8 % self.dst_width
                        src_c = (index_cm - dst_c_st) % 8 * 2 + 1 + dst_c_st
                        src_hw = (index_cm - dst_c_st) // 8 * 8
                        total_outgm_size = self.dst_channel * self.dst_height \
                                           * self.dst_width
                        move_size = dst_c * self.dst_height * self.dst_width \
                                    + dst_h * self.dst_width + dst_w
                        dst_channel_one_loop = \
                            self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(move_size \
                                                        + (dst_hw_nbrust - 1) * 2 \
                                                        * 8 >= total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(
                                dst_hw_nbrust - num_sub)

                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    dst_n * self.chw +
                                    dst_c * self.hw +
                                    dst_h * self.w +
                                    dst_w],
                                output_ub_fp32[
                                    src_c,
                                    src_hw],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

                with self.tik_instance.for_range(dst_c_nbrust - 1, dst_c_nbrust) \
                        as index_c:
                    with self.tik_instance.for_range(0, dst_hw_nbrust) \
                            as index_hwm:
                        src_list = [input_ub_fp32[index_c, i + index_hwm * 16, 0]
                                    for i in range(0, 16)]
                        dist_list = [output_ub_fp32[i + index_c * 16, index_hwm * 16]
                                     for i in range(0, 16)]
                        self.tik_instance.vnchwconv(
                            False,
                            False,
                            dist_list,
                            src_list,
                            2,
                            1,
                            1)

                    dst_c_st = index_c * 16
                    dst_c_end = self.dst_channel

                    with self.tik_instance.for_range(dst_c_st, dst_c_end) \
                            as index_cm:
                        dst_n = index_n
                        dst_c = index_cm
                        dst_h = 0
                        dst_w = 0
                        src_c = (index_cm - dst_c_st) % 8 * 2 + dst_c_st
                        src_hw = (index_cm - dst_c_st) // 8 * 8
                        total_outgm_size = self.dst_channel \
                                           * self.dst_height * self.dst_width
                        move_size = dst_c * self.dst_height * self.dst_width \
                                    + dst_h * self.dst_width + dst_w
                        dst_channel_one_loop = \
                            self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 \
                                * 8 >= total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(
                                dst_hw_nbrust - num_sub)

                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    dst_n * self.chw +
                                    dst_c * self.hw +
                                    dst_h * self.w +
                                    dst_w],
                                output_ub_fp32[
                                    src_c,
                                    src_hw],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

                        dst_n = index_n
                        dst_c = index_cm
                        dst_h = 8 // self.dst_width
                        dst_w = 8 % self.dst_width
                        src_c = (index_cm - dst_c_st) % 8 * 2 + 1 + dst_c_st
                        src_hw = (index_cm - dst_c_st) // 8 * 8
                        total_outgm_size = self.dst_channel \
                                           * self.dst_height * self.dst_width
                        move_size = dst_c * self.dst_height * self.dst_width \
                                    + dst_h * self.dst_width + dst_w
                        dst_channel_one_loop = \
                            self.tik_instance.Scalar("int32")
                        dst_channel_one_loop.set_as(dst_hw_nbrust)

                        with self.tik_instance.if_scope(
                                move_size + (dst_hw_nbrust - 1) * 2 \
                                * 8 >= total_outgm_size):
                            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                                       * 8 - total_outgm_size + 16) // 16
                            dst_channel_one_loop.set_as(
                                dst_hw_nbrust - num_sub)

                        with self.tik_instance.if_scope(
                                dst_channel_one_loop > 0):
                            self.tik_instance.data_move(
                                self.dst_image[
                                    dst_n * self.chw +
                                    dst_c * self.hw +
                                    dst_h * self.w +
                                    dst_w],
                                output_ub_fp32[
                                    src_c,
                                    src_hw],
                                0,
                                dst_channel_one_loop,
                                1,
                                1,
                                1)

    # ’pylint: disable=too-many-locals,too-many-statements,too-many-arguments
    # ’pylint: disable=invalid-name
    def _move_slice(self, dst_n, dst_c, dst_h, dst_w,
                    src_c, src_hw,
                    dst_hw_nbrust, output_ub_fp32):
        """
        Move data from output ub to gm

        Parameters
        ----------
        dst_n : int
            n index of output tensor
        dst_c : int
            c index of output tensor
        dst_h : int
            h index of output tensor
        dst_w : int
            w index of output tensor
        src_c : int
            c index of output tensor on ub
        src_hw : int
            hw index of output tensor on ub
        dst_hw_nbrust : int
            nbrust of output tensor to move
        output_ub_fp32 : ub_buf
            output tensor
        """

        total_outgm_size = self.dst_batch * self.dst_channel \
                           * self.dst_height * self.dst_width
        move_size = dst_n * self.dst_channel * self.dst_height * self.dst_width \
                    + dst_c * self.dst_height * self.dst_width \
                    + dst_h * self.dst_width + dst_w
        dst_channel_one_loop = self.tik_instance.Scalar("int32")
        dst_channel_one_loop.set_as(dst_hw_nbrust)

        with self.tik_instance.if_scope(move_size + (dst_hw_nbrust - 1) * 2 * 8
                                        >= total_outgm_size):
            num_sub = (move_size + (dst_hw_nbrust - 1) * 2 \
                       * 8 - total_outgm_size + 15) // 16
            dst_channel_one_loop.set_as(dst_hw_nbrust - num_sub)

        with self.tik_instance.if_scope(dst_channel_one_loop > 0):
            self.tik_instance.data_move(
                self.dst_image[
                    dst_n * self.chw +
                    dst_c * self.hw +
                    dst_h * self.w +
                    dst_w],
                output_ub_fp32[
                    src_c,
                    src_hw],
                0,
                dst_channel_one_loop,
                1,
                1,
                1)

    # ’pylint: disable=unused-argument
    def _chk_mov(self, dst_c_st, dst_c_end,
                 index_n, index_cm, dst_hw_nbrust, output_ub_fp32):
        """
        Calculate start point and length for data move
        in case overlap global memory

        Parameters
        ----------
        dst_c_st : int
            c1 start point at c1 dimension
        dst_c_end : int
            c1 end point at c1 dimension
        index_n : int
            n index of input tensor
        index_cm : int
            current c index for data move
        dst_hw_nbrust :int
            nburst of hw size
        output_ub_fp32 : ub_buf
            output tensor
        """

        dst_n = index_n
        dst_c = index_cm
        dst_h = 0
        dst_w = 0
        src_c = (index_cm - dst_c_st) % 8 * 2 + dst_c_st
        src_hw = (index_cm - dst_c_st) // 8 * 8
        self._move_slice(dst_n, dst_c, dst_h, dst_w, src_c,
                         src_hw, dst_hw_nbrust, output_ub_fp32)

        dst_n = index_n
        dst_c = index_cm
        dst_h = 8 // self.dst_width
        dst_w = 8 % self.dst_width
        src_c = (index_cm - dst_c_st) % 8 * 2 + 1 + dst_c_st
        src_hw = (index_cm - dst_c_st) // 8 * 8
        self._move_slice(dst_n, dst_c, dst_h, dst_w,
                         src_c, src_hw, dst_hw_nbrust, output_ub_fp32)


# ’pylint:disable=unused-argument
def five_2_four_v200_fp32fp16(src_dict, dst_dict,
                              src_format, dst_format,
                              kernel_name="five_2_four"):
    """
    algorithm: five_2_four for V200
    calculating: change data format from NC1HWC0 to NCHW

    Parameters
    ----------
    src: dict
        contains shape and dtype information of input tensor
    dst: dict
        contains shape and dtype information of output tensor
    src_format: str
        represents the format of input tensor, only support "NC1HWC0"
    dst_format: str
        represents the format of output tensor, only support "NCHW"
    kernel_name: str
        cce kernel name, default value is "five_2_four"

    Returns
    -------
    None
    """

    obj = VnchwConv(src_dict, dst_dict, kernel_name)
    obj.compute_slice()
    tik_instance = obj.tik_instance
    tik_instance.BuildCCE(kernel_name=obj.kernel_name,
                          inputs=[obj.src_image], outputs=[obj.dst_image])

    return tik_instance
