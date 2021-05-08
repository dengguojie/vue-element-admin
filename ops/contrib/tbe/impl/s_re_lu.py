"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

s_relu
"""

from te import tik
import numpy as np

UB_BUFF_MAX = 240 * 1024


class SRelu:
    def _aicore_in_use_select(self, length):
        self.xlen_each_core = (length + self.aicore_use - 1) // self.aicore_use
        self.xlen_last_core = length - self.xlen_each_core * (self.aicore_use - 1)
        self.aicore_use = (length + self.xlen_each_core - 1) // self.xlen_each_core
        if (self.aicore_use == 1):
            self.xlen_last_core = self.xlen_each_core
        print("self.xlen_each_core:", self.xlen_each_core, "self.xlen_last_core:", self.xlen_last_core)
        print("self.aicore_use:", self.aicore_use)

    def __init__(self, shape, channel_shared, kern_name):
        in_n = shape[0]
        in_c1 = shape[1]
        in_h = shape[2]
        in_w = shape[3]
        in_c = in_c1 * 16
        in_hw = in_h * in_w

        if channel_shared:
            raise RuntimeError("not support channel_shared")

        # Consider that hw was too bigger to make ub space overflow
        max_hw = (UB_BUFF_MAX - 16 * in_c1 * 2 * 4) // (3 * 16 * 2)
        if (max_hw <= 0):
            raise RuntimeError("dim c can not bigger than", (UB_BUFF_MAX - 96) // 8)

        self.in_n = in_n
        self.in_c = in_c
        self.in_h = in_h
        self.in_w = in_w
        self.in_c1 = in_c1
        self.in_hw = in_hw
        self.in_channel_shared = channel_shared
        self.kern_name = kern_name

        self.aicore_use = 2
        self._aicore_in_use_select(self.in_hw)

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.gm_input = self.tik_instance.Tensor("float16", (in_n, in_c1, in_h, in_w, 16), name="gm_input",
                                                 scope=tik.scope_gm)
        self.gm_output = self.tik_instance.Tensor("float16", (in_n, in_c1, in_h, in_w, 16), name="gm_output",
                                                  scope=tik.scope_gm)
        self.gm_tr = self.tik_instance.Tensor("float16", (1, in_c1, 1, 1, 16), name="gm_tr", scope=tik.scope_gm)
        self.gm_ar = self.tik_instance.Tensor("float16", (1, in_c1, 1, 1, 16), name="gm_ar", scope=tik.scope_gm)
        self.gm_tl = self.tik_instance.Tensor("float16", (1, in_c1, 1, 1, 16), name="gm_tl", scope=tik.scope_gm)
        self.gm_al = self.tik_instance.Tensor("float16", (1, in_c1, 1, 1, 16), name="gm_al", scope=tik.scope_gm)

        self.max_hw = max_hw
        if (max_hw >= self.xlen_each_core):
            self.max_hw = self.xlen_each_core
            self.in_cycle_hw = 1
        else:
            self.in_cycle_hw = (self.xlen_each_core + max_hw - 1) // max_hw

        print("(n,c,h,w):", self.in_n, self.in_c, self.in_h, self.in_w)
        print("hw:", self.in_hw, "channel_shared:", self.in_channel_shared)
        print("max_hw:", self.max_hw, "cycle_hw:", self.in_cycle_hw)

    def tiling_mode_select(self):
        self.mode = 1

    def global_init(self):
        pass

    def mode1_compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as index:
            with self.tik_instance.if_scope(index != self.aicore_use - 1):
                self._mode1_compute_each_core(self.xlen_each_core, (index * self.xlen_each_core))
            with self.tik_instance.else_scope():
                self._mode1_compute_each_core(self.xlen_last_core, (index * self.xlen_each_core))

        self.tik_instance.BuildCCE(kernel_name=self.kern_name,
                                   inputs=[self.gm_input, self.gm_tr, self.gm_ar, self.gm_tl, self.gm_al],
                                   outputs=[self.gm_output])

    def _calc_s_relu_step1(self, cycle1, i_in_c1):
        with self.tik_instance.if_scope(cycle1 != 0):
            with self.tik_instance.for_range(0, cycle1) as i_cycle1:
                offset = 128 * 255 * i_cycle1
                # use 4 steps to get SRelu result.
                # 1. pick out src > tr
                # 2. pick out src < tl
                # 3. pick out tl < src < tr
                # 4. add above 3 steps result

                # if src > tr then dst equal to  ar * (src - tr); else then dst equal to 0
                self.tik_instance.vmax(128, self.ub_out[offset], self.ub_step[offset],
                                       self.ub_tr[0, i_in_c1, 0, 0, 0], self.max_times, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vsub(128, self.ub_out[offset], self.ub_out[offset],
                                       self.ub_tr[0, i_in_c1, 0, 0, 0], self.max_times, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vmul(128, self.ub_out[offset], self.ub_out[offset],
                                       self.ub_ar[0, i_in_c1, 0, 0, 0], self.max_times, 1, 1, 0, 8, 8, 0)
                # if src < tl then dst equal to al * (src - tl); else then dst = 0
                self.tik_instance.vmin(128, self.ub_tmpout[offset], self.ub_step[offset],
                                       self.ub_tl[0, i_in_c1, 0, 0, 0], self.max_times, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vsub(128, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                       self.ub_tl[0, i_in_c1, 0, 0, 0], self.max_times, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vmul(128, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                       self.ub_al[0, i_in_c1, 0, 0, 0], self.max_times, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vadd(128, self.ub_out[offset], self.ub_out[offset],
                                       self.ub_tmpout[offset], self.max_times, 1, 1, 1, 8, 8, 8)
                # if src > tr then dst equal to tr;
                # else if src < tl src then dst equal to tl
                # else then dst equal to src
                self.tik_instance.vmin(128, self.ub_tmpout[offset], self.ub_step[offset],
                                       self.ub_tr[0, i_in_c1, 0, 0, 0], self.max_times, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vmax(128, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                       self.ub_tl[0, i_in_c1, 0, 0, 0], self.max_times, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vadd(128, self.ub_out[offset], self.ub_out[offset],
                                       self.ub_tmpout[offset], self.max_times, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.else_scope():
            pass

    def _calc_s_relu_step2(self, offset, i_in_c1, remain1):
        with self.tik_instance.if_scope(remain1 != 0):
            self.tik_instance.vmax(128, self.ub_out[offset], self.ub_step[offset],
                                   self.ub_tr[0, i_in_c1, 0, 0, 0], remain1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vsub(128, self.ub_out[offset], self.ub_out[offset],
                                   self.ub_tr[0, i_in_c1, 0, 0, 0], remain1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmul(128, self.ub_out[offset], self.ub_out[offset],
                                   self.ub_ar[0, i_in_c1, 0, 0, 0], remain1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmin(128, self.ub_tmpout[offset], self.ub_step[offset],
                                   self.ub_tl[0, i_in_c1, 0, 0, 0], remain1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vsub(128, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                   self.ub_tl[0, i_in_c1, 0, 0, 0], remain1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmul(128, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                   self.ub_al[0, i_in_c1, 0, 0, 0], remain1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vadd(128, self.ub_out[offset], self.ub_out[offset], self.ub_tmpout[offset],
                                   remain1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmin(128, self.ub_tmpout[offset], self.ub_step[offset],
                                   self.ub_tr[0, i_in_c1, 0, 0, 0], remain1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vmax(128, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                   self.ub_tl[0, i_in_c1, 0, 0, 0], remain1, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vadd(128, self.ub_out[offset], self.ub_out[offset], self.ub_tmpout[offset],
                                   remain1, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.else_scope():
            pass

    def _calc_s_relu_step3(self, offset, i_in_c1, remain2):
        with self.tik_instance.if_scope(remain2 != 0):
            self.tik_instance.vmax(16, self.ub_out[offset], self.ub_step[offset],
                                   self.ub_tr[0, i_in_c1, 0, 0, 0], remain2, 1, 1, 0, 1, 1, 0)
            self.tik_instance.vsub(16, self.ub_out[offset], self.ub_out[offset],
                                   self.ub_tr[0, i_in_c1, 0, 0, 0], remain2, 1, 1, 0, 1, 1, 0)
            self.tik_instance.vmul(16, self.ub_out[offset], self.ub_out[offset],
                                   self.ub_ar[0, i_in_c1, 0, 0, 0], remain2, 1, 1, 0, 1, 1, 0)
            self.tik_instance.vmin(16, self.ub_tmpout[offset], self.ub_step[offset],
                                   self.ub_tl[0, i_in_c1, 0, 0, 0], remain2, 1, 1, 0, 1, 1, 0)
            self.tik_instance.vsub(16, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                   self.ub_tl[0, i_in_c1, 0, 0, 0], remain2, 1, 1, 0, 1, 1, 0)
            self.tik_instance.vmul(16, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                   self.ub_al[0, i_in_c1, 0, 0, 0], remain2, 1, 1, 0, 1, 1, 0)
            self.tik_instance.vadd(16, self.ub_out[offset], self.ub_out[offset], self.ub_tmpout[offset],
                                   remain2, 1, 1, 1, 1, 1, 1)

            self.tik_instance.vmin(16, self.ub_tmpout[offset], self.ub_step[offset],
                                   self.ub_tr[0, i_in_c1, 0, 0, 0], remain2, 1, 1, 0, 1, 1, 0)
            self.tik_instance.vmax(16, self.ub_tmpout[offset], self.ub_tmpout[offset],
                                   self.ub_tl[0, i_in_c1, 0, 0, 0], remain2, 1, 1, 0, 1, 1, 0)
            self.tik_instance.vadd(16, self.ub_out[offset], self.ub_out[offset], self.ub_tmpout[offset],
                                   remain2, 1, 1, 1, 1, 1, 1)
        with self.tik_instance.else_scope():
            pass

    def _calc_cur_hw(self, i_cycle_hw, cycle_hw, length, max_hw):
        with self.tik_instance.if_scope(i_cycle_hw != (cycle_hw - 1)):
            self.in_cur_hw.set_as(max_hw)
        with self.tik_instance.else_scope():
            self.in_cur_hw.set_as(length - (max_hw * i_cycle_hw))

    def _calc_max_times(self, cycle1):
        with self.tik_instance.if_scope(cycle1 != 0):
            self.max_times.set_as(255)
        with self.tik_instance.else_scope():
            self.max_times.set_as(0)

    def _mode1_compute_each_core(self, length, offset):
        channel_shared = self.in_channel_shared
        cycle_hw = self.in_cycle_hw
        max_hw = self.max_hw
        in_c1 = self.in_c1
        num_n = self.in_n
        num_w = self.in_w

        self.ub_tr = self.tik_instance.Tensor("float16", (1, in_c1, 1, 1, 16), name="ub_tr", scope=tik.scope_ubuf)
        self.ub_ar = self.tik_instance.Tensor("float16", (1, in_c1, 1, 1, 16), name="ub_ar", scope=tik.scope_ubuf)
        self.ub_tl = self.tik_instance.Tensor("float16", (1, in_c1, 1, 1, 16), name="ub_tl", scope=tik.scope_ubuf)
        self.ub_al = self.tik_instance.Tensor("float16", (1, in_c1, 1, 1, 16), name="ub_al", scope=tik.scope_ubuf)
        self.ub_step = self.tik_instance.Tensor("float16", (1, 1, max_hw, 16), name="ub_step", scope=tik.scope_ubuf)
        self.ub_out = self.tik_instance.Tensor("float16", (1, 1, max_hw, 16), name="ub_out", scope=tik.scope_ubuf)
        self.ub_tmpout = self.tik_instance.Tensor("float16", (1, 1, max_hw, 16), name="ub_tmpout",
                                                  scope=tik.scope_ubuf)
        self.in_cur_hw = self.tik_instance.Scalar("uint32")
        self.max_times = self.tik_instance.Scalar("uint16")

        if not channel_shared:
            self.tik_instance.data_move(self.ub_tr, self.gm_tr, 0, 1, in_c1, 0, 0, 0)
            self.tik_instance.data_move(self.ub_ar, self.gm_ar, 0, 1, in_c1, 0, 0, 0)
            self.tik_instance.data_move(self.ub_tl, self.gm_tl, 0, 1, in_c1, 0, 0, 0)
            self.tik_instance.data_move(self.ub_al, self.gm_al, 0, 1, in_c1, 0, 0, 0)
        # else not support now, see also __init__()

        with self.tik_instance.for_range(0, num_n) as i_n:
            with self.tik_instance.for_range(0, in_c1) as i_in_c1:
                with self.tik_instance.for_range(0, cycle_hw) as i_cycle_hw:
                    self._calc_cur_hw(i_cycle_hw, cycle_hw, length, max_hw)

                    hoffset = (offset + i_cycle_hw * max_hw) // num_w
                    woffset = (offset + i_cycle_hw * max_hw) - hoffset * num_w
                    self.tik_instance.data_move(self.ub_step, self.gm_input[i_n, i_in_c1, hoffset, woffset, 0],
                                                0, 1, self.in_cur_hw, 0, 0, 0)
                    cycle1 = self.in_cur_hw // (8 * 255)
                    remain1 = (self.in_cur_hw - cycle1 * 8 * 255) // 8
                    remain2 = (self.in_cur_hw - cycle1 * 8 * 255) - remain1 * 8
                    self._calc_max_times(cycle1)
                    self._calc_s_relu_step1(cycle1, i_in_c1)
                    offset = 128 * 255 * cycle1
                    self._calc_s_relu_step2(offset, i_in_c1, remain1)
                    offset = offset + 128 * remain1
                    self._calc_s_relu_step3(offset, i_in_c1, remain2)

                    self.tik_instance.data_move(self.gm_output[i_n, i_in_c1, hoffset, woffset, 0], self.ub_out,
                                                0, 1, self.in_cur_hw, 0, 0, 0)

    def tik_output_debug(self):
        data = np.ones([self.in_n, (self.in_c + 15) // 16, self.in_h, self.in_w, 16]).astype(np.float16)
        data = data * 0.7
        t_r = np.ones([1, (self.in_c + 15) // 16, 1, 1, 16]).astype(np.float16)
        a_r = np.ones([1, (self.in_c + 15) // 16, 1, 1, 16]).astype(np.float16)
        t_l = np.ones([1, (self.in_c + 15) // 16, 1, 1, 16]).astype(np.float16)
        a_l = np.ones([1, (self.in_c + 15) // 16, 1, 1, 16]).astype(np.float16)
        feed_dict = {
            "gm_input": data,
            "gm_tr": t_r,
            "gm_ar": a_r,
            "gm_tl": t_l,
            "gm_al": a_l,
        }
        out = self.tik_instance.tikdb.start_debug(feed_dict, False)
        print(out)


def s_re_lu(input_x, t_r, a_r, t_l, a_l, output_y, channel_shared, kernel_name="s_re_lu", test=False):
    """
    piecewise linear activation function
    :param input_x: input
    :param t_r: t_r
    :param a_r: a_r
    :param t_l: t_l
    :param a_l: a_l
    :param y: output
    :param channel_shared:
    :param kernel_name:kernel_name
    :param test: for debug
    :constrictions:
        1) channel_shared must be false
        2) Dim(c) of input_x must be less than 30700
    """
    shape = input_x.get("shape")
    obj = SRelu(shape, channel_shared, kernel_name)

    obj.tiling_mode_select()
    if obj.mode == 0:
        raise RuntimeError("can not select a valid tiling mode.")

    obj.global_init()

    switch = {
        1: obj.mode1_compute
    }

    switch[obj.mode]()
    if not test:
        return 0

    obj.tik_output_debug()
    return 0
