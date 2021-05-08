# coding: utf-8
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

"""
from te import tik

G_UBMAX_BYTE = 240 * 1024
G_L1MAX_BYTE = 1024 * 1024
G_BLOCK_BYTE = 32
G_BLOCK_NUM_PER_CMD = 8
G_CMD_MAX_REPEAT = 255
G_SIZE_MAP = {"float16": 2, "float32": 4, "int32": 4}


class Utils:

    def __init__(self, tik_inst, dtype):
        self.tik_inst = tik_inst
        self.num_per_block = G_BLOCK_BYTE // G_SIZE_MAP.get(dtype)
        self.max_num_per_cmd = G_BLOCK_NUM_PER_CMD * self.num_per_block
        self.max_num_per_loop = G_CMD_MAX_REPEAT * self.max_num_per_cmd

    def auto_vconv(self, dtype, length, dst_ub, src_ub, dst_buf_idx, src0_buf_idx):
        if dtype == "16T32":
            dst_d, src_d = 8, 4
        else:
            dst_d, src_d = 4, 8

        repeat_time = length // 64
        repeat_time_last = length % 64
        if repeat_time > 0:
            self.tik_inst.vconv(64, '', dst_ub[dst_buf_idx], src_ub[src0_buf_idx], repeat_time, 1,
                                1, dst_d, src_d)
        if repeat_time_last > 0:
            self.tik_inst.vconv(repeat_time_last, '', dst_ub[dst_buf_idx + repeat_time * 64],
                                src_ub[src0_buf_idx + repeat_time * 64], 1, 1, 1, dst_d, src_d)

    def _parse_cmd(self, cmd, scalar, src0_buf, src1_buf, dst_buf, mask, repeat, rep_stride):
        if (cmd == "vadds"):
            self.tik_inst.vadds(mask, dst_buf, src0_buf, scalar, repeat, 1, 1, rep_stride,
                                rep_stride)
        elif (cmd == "vmul"):
            self.tik_inst.vmul(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1, 1, rep_stride,
                               rep_stride, rep_stride)
        elif (cmd == "vadd"):
            self.tik_inst.vadd(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1, 1, rep_stride,
                               rep_stride, rep_stride)
        elif (cmd == "vmuls"):
            self.tik_inst.vmuls(mask, dst_buf, src0_buf, scalar, repeat, 1, 1, rep_stride,
                                rep_stride)
        elif (cmd == "vsub"):
            self.tik_inst.vsub(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1, 1, rep_stride,
                               rep_stride, rep_stride)
        elif (cmd == "vabs"):
            self.tik_inst.vabs(mask, dst_buf, src0_buf, repeat, 1, 1, rep_stride, rep_stride)
        else:
            raise RuntimeError("Not supported cmd:{}".format(cmd))

    def cmd_vec(self, cmd, num, dst_buf, src0_buf, src1_buf, dst_buf_idx, src0_buf_idx,
                src1_buf_idx, scalar):
        loop = num // self.max_num_per_loop
        if loop > 0:
            with self.tik_inst.for_range(0, loop) as loop_i:
                offset = loop_i * self.max_num_per_loop
                self._parse_cmd(cmd, scalar, src0_buf[offset + src0_buf_idx],
                                src1_buf[offset + src1_buf_idx], dst_buf[offset + dst_buf_idx],
                                self.max_num_per_cmd, G_CMD_MAX_REPEAT, 8)

        cur_offset = loop * self.max_num_per_loop
        repeat = (num - cur_offset) // self.max_num_per_cmd
        if repeat > 0:
            self._parse_cmd(cmd, scalar, src0_buf[cur_offset + src0_buf_idx],
                            src1_buf[cur_offset + src1_buf_idx], dst_buf[cur_offset + dst_buf_idx],
                            self.max_num_per_cmd, repeat, 8)

        cur_offset = cur_offset + repeat * self.max_num_per_cmd
        remain = num - cur_offset
        if remain > 0:
            self._parse_cmd(cmd, scalar, src0_buf[cur_offset + src0_buf_idx],
                            src1_buf[cur_offset + src1_buf_idx], dst_buf[cur_offset + dst_buf_idx],
                            remain, 1, 0)


class InterpV2():
    """
    image interpolation using resize bilinear
    Parameters
    ----------
    input0 : Image to be interpolated, an NC1HWC0 tensor, float16
    input1 : Define output shape, an NC1HWC0 tensor, float16
    output0 : Image after interpolate, an NC1HWC0 tensor, float16
    kernel_name: str, default is "Interpv2"

    Returns
    -------
    None
    """

    def __init__(self, input0, input1, output0, kernel_name):
        self.shape0 = input0.get("ori_shape")
        self.out_shape0 = input1.get("ori_shape")
        self.kernel_name = kernel_name

        if len(self.shape0) != 4 or len(self.out_shape0) != 4:
            raise RuntimeError("ori_shape of input should be 4D")
        if self.shape0[0:2] != self.out_shape0[0:2]:
            raise RuntimeError("dim in n and c of two input should be same")

        self.input_n, self.input_c, self.input_h, self.input_w = self.shape0
        self.output_n, self.output_c, self.output_h, self.output_w = self.out_shape0
        if (self.input_n * self.input_c) % 128 != 0 and self.input_n * self.input_c > 128:
            raise RuntimeError("the product of dim of N and dim of C shall be less \
                               than 128 if is not aligned with 128 ")
        if self.input_h > 64 or self.input_w > 64 or self.output_h > 64 or self.output_w > 64:
            raise RuntimeError("h and w in two input should not bigger than 64")

        self.tik_inst = tik.Tik(tik.Dprofile('v100', 'mini'))
        self.utilfp16 = Utils(self.tik_inst, "float16")

        if self.output_h > 1:
            self.height_scale = float(self.input_h - 1) / (self.output_h - 1)
        else:
            self.height_scale = float(self.input_h) / (self.output_h)
        if self.output_w > 1:
            self.width_scale = float(self.input_w - 1) / (self.output_w - 1)
        else:
            self.width_scale = float(self.input_w) / (self.output_w)

        self.data_a_gm = self.tik_inst.Tensor("float16", (
            self.input_n, self.input_c // 16, self.input_h, self.input_w, 16), tik.scope_gm,
                                              "data_a_gm")
        self.data_b_gm = self.tik_inst.Tensor("float16", (
            self.output_n, self.output_c // 16, self.output_h, self.output_w, 16), tik.scope_gm,
                                              "data_b_gm")
        self.data_b_ub = self.tik_inst.Tensor("float16", (1, 1, 1, 1, 16), tik.scope_ubuf,
                                              "data_b_ub")

        self.dst_gm = self.tik_inst.Tensor("float16", (
            self.output_n, self.output_c // 16, self.output_h, self.output_w, 16), tik.scope_gm,
                                           "dst_gm")
        self._memory_appli(self.output_n, self.output_c, self.output_h, self.output_w)

    def compute(self):
        self.tik_inst.data_move(self.data_b_ub, self.data_b_gm, 0, 1, 1, 0, 0)
        self._cal_weight(self.output_h, self.hin_int_ub, self.hin_float_ub, self.h_bottom_ub,
                         self.h_top_ub, self.height_scale)
        self._cal_weight(self.output_w, self.win_int_ub, self.win_float_ub, self.w_right_ub,
                         self.w_left_ub, self.width_scale)
        self._cal_interp(self.h_in, self.h1p, self.w_in, self.w1p, self.h_top_value,
                         self.h_bottom_value, self.num_br, self.num_bl, self.num_tr, self.num_tl)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.data_a_gm, self.data_b_gm],
                               outputs=[self.dst_gm], enable_l2=True)
        return self.tik_inst

    def _memory_appli(self, output_n, output_c, output_h, output_w):
        self.hin_int_ub = self.tik_inst.Tensor("int32", (1, output_h), tik.scope_ubuf, "hin_int_ub")
        self.hin_float_ub = self.tik_inst.Tensor("float16", (1, output_h), tik.scope_ubuf,
                                                 "hin_float_ub")
        self.h_bottom_ub = self.tik_inst.Tensor("float16", (1, output_h), tik.scope_ubuf,
                                                "h_bottom_ub")
        self.h_top_ub = self.tik_inst.Tensor("float16", (1, output_h), tik.scope_ubuf, "h_top_ub")

        self.win_int_ub = self.tik_inst.Tensor("int32", (1, output_w), tik.scope_ubuf, "win_int_ub")
        self.win_float_ub = self.tik_inst.Tensor("float16", (1, output_w), tik.scope_ubuf,
                                                 "win_float_ub")
        self.w_right_ub = self.tik_inst.Tensor("float16", (1, output_w), tik.scope_ubuf,
                                               "w_right_ub")
        self.w_left_ub = self.tik_inst.Tensor("float16", (1, output_w), tik.scope_ubuf, "w_left_ub")

        self.lerp_br = self.tik_inst.Tensor("float16", (1, output_w), tik.scope_ubuf, "lerp_br")
        self.lerp_tr = self.tik_inst.Tensor("float16", (1, output_w), tik.scope_ubuf, "lerp_tr")
        self.lerp_tl = self.tik_inst.Tensor("float16", (1, output_w), tik.scope_ubuf, "lerp_tl")
        self.lerp_bl = self.tik_inst.Tensor("float16", (1, output_w), tik.scope_ubuf, "lerp_bl")

        self.vector_locatevalue_br = self.tik_inst.Tensor("float16", (1, output_n * output_c),
                                                          tik.scope_ubuf, "vector_locatevalue_br")
        self.vector_locatevalue_bl = self.tik_inst.Tensor("float16", (1, output_n * output_c),
                                                          tik.scope_ubuf, "vector_locatevalue_bl")
        self.vector_locatevalue_tr = self.tik_inst.Tensor("float16", (1, output_n * output_c),
                                                          tik.scope_ubuf, "vector_locatevalue_tr")
        self.vector_locatevalue_tl = self.tik_inst.Tensor("float16", (1, output_n * output_c),
                                                          tik.scope_ubuf, "vector_locatevalue_tl")
        self.num_br = self.tik_inst.Scalar("float16")
        self.num_bl = self.tik_inst.Scalar("float16")
        self.num_tr = self.tik_inst.Scalar("float16")
        self.num_tl = self.tik_inst.Scalar("float16")
        self.h_in = self.tik_inst.Scalar("int32")
        self.h1p = self.tik_inst.Scalar("int32")
        self.h_bottom_value = self.tik_inst.Scalar("float16")
        self.h_top_value = self.tik_inst.Scalar("float16")
        self.w_in = self.tik_inst.Scalar("int32")
        self.w1p = self.tik_inst.Scalar("int32")

    def _cal_weight(self, cal_length, inint_ub, infloat_ub, br_ub, tl_ub, scale):
        with self.tik_inst.for_range(0, cal_length)as i_out:
            inint_ub[0, i_out] = i_out
        self.tik_inst.vconv(cal_length, "", infloat_ub, inint_ub, 1, 1, 1, 4, 8, 1.0)
        self.tik_inst.vmuls(cal_length, infloat_ub, infloat_ub, scale, 1, 1, 1, 8, 8, 0)
        self.tik_inst.vconv(cal_length, "floor", inint_ub, infloat_ub, 1, 1, 1, 8, 4)
        self.tik_inst.vconv(cal_length, "", br_ub, inint_ub, 1, 1, 1, 4, 8, 1.0)
        self.tik_inst.vsub(cal_length, br_ub, infloat_ub, br_ub, 1, 1, 1, 1, 8, 8, 8, 0)
        self.tik_inst.vadds(cal_length, tl_ub, br_ub, -1.0, 1, 1, 1, 8, 8, 0)
        self.tik_inst.vabs(cal_length, tl_ub, tl_ub, 1, 1, 1, 8, 8, 0)

    def _cal_interp(self, h_in, h1p, w_in, w1p, h_top_value, h_bottom_value,
                    num_br, num_bl, num_tr, num_tl):
        with self.tik_inst.for_range(0, self.output_h) as h_out:
            h_in.set_as(self.hin_int_ub[0, h_out])
            with self.tik_inst.if_scope(h_in < self.input_h - 1):
                h1p.set_as(1)
            with self.tik_inst.else_scope():
                h1p.set_as(0)
            h_top_value.set_as(self.h_top_ub[0, h_out])
            h_bottom_value.set_as(self.h_bottom_ub[0, h_out])
            self.utilfp16.cmd_vec("vmuls", self.output_w, self.lerp_tr, self.w_right_ub,
                                  self.w_right_ub, 0, 0, 0, h_top_value)
            self.utilfp16.cmd_vec("vmuls", self.output_w, self.lerp_br, self.w_right_ub,
                                  self.w_right_ub, 0, 0, 0, h_bottom_value)
            self.utilfp16.cmd_vec("vmuls", self.output_w, self.lerp_tl, self.w_left_ub,
                                  self.w_left_ub, 0, 0, 0, h_top_value)
            self.utilfp16.cmd_vec("vmuls", self.output_w, self.lerp_bl, self.w_left_ub,
                                  self.w_left_ub, 0, 0, 0, h_bottom_value)
            with self.tik_inst.for_range(0, self.output_w) as w_out:
                w_in.set_as(self.win_int_ub[0, w_out])
                with self.tik_inst.if_scope(w_in < self.input_w - 1):
                    w1p.set_as(1)
                with self.tik_inst.else_scope():
                    w1p.set_as(0)
                self.tik_inst.data_move(self.vector_locatevalue_br, self.data_a_gm[0, 0, h_in + h1p, w_in + w1p, 0],
                                        0, self.input_n * self.input_c // 16, 1, self.input_h * self.input_w - 1, 0)
                self.tik_inst.data_move(self.vector_locatevalue_bl, self.data_a_gm[0, 0, h_in + h1p, w_in, 0],
                                        0, self.input_n * self.input_c // 16, 1, self.input_h * self.input_w - 1, 0)
                self.tik_inst.data_move(self.vector_locatevalue_tr, self.data_a_gm[0, 0, h_in, w_in + w1p, 0], 0,
                                        self.input_n * self.input_c // 16, 1, self.input_h * self.input_w - 1, 0)
                self.tik_inst.data_move(self.vector_locatevalue_tl, self.data_a_gm[0, 0, h_in, w_in, 0], 0,
                                        self.input_n * self.input_c // 16, 1, self.input_h * self.input_w - 1, 0)
                num_br.set_as(self.lerp_br[0, w_out])
                num_bl.set_as(self.lerp_bl[0, w_out])
                num_tr.set_as(self.lerp_tr[0, w_out])
                num_tl.set_as(self.lerp_tl[0, w_out])
                self.utilfp16.cmd_vec("vmuls", self.output_n * self.output_c, self.vector_locatevalue_br,
                                      self.vector_locatevalue_br, self.vector_locatevalue_br, 0, 0, 0, num_br)
                self.utilfp16.cmd_vec("vmuls", self.output_n * self.output_c, self.vector_locatevalue_bl,
                                      self.vector_locatevalue_bl, self.vector_locatevalue_bl, 0, 0, 0, num_bl)
                self.utilfp16.cmd_vec("vmuls", self.output_n * self.output_c, self.vector_locatevalue_tr,
                                      self.vector_locatevalue_tr, self.vector_locatevalue_tr, 0, 0, 0, num_tr)
                self.utilfp16.cmd_vec("vmuls", self.output_n * self.output_c, self.vector_locatevalue_tl,
                                      self.vector_locatevalue_tl, self.vector_locatevalue_tl, 0, 0, 0, num_tl)
                self.utilfp16.cmd_vec("vadd", self.output_n * self.output_c, self.vector_locatevalue_br,
                                      self.vector_locatevalue_br, self.vector_locatevalue_bl, 0, 0, 0, 0)
                self.utilfp16.cmd_vec("vadd", self.output_n * self.output_c, self.vector_locatevalue_br,
                                      self.vector_locatevalue_br, self.vector_locatevalue_tr, 0, 0, 0, 0)
                self.utilfp16.cmd_vec("vadd", self.output_n * self.output_c, self.vector_locatevalue_br,
                                      self.vector_locatevalue_br, self.vector_locatevalue_tl, 0, 0, 0, 0)
                self.tik_inst.data_move(self.dst_gm[0, 0, h_out, w_out, 0], self.vector_locatevalue_br, 0,
                                        self.output_n * self.output_c // 16, 1, 0, self.output_h * self.output_w - 1)

    def tiling_mode_select(self):
        self.mode = 1

    def global_init(self):
        pass


def interpv2(input0, input1, output, kernel_name="Interpv2"):
    obj = InterpV2(input0, input1, output, kernel_name)
    obj.tiling_mode_select()
    if obj.mode == 0:
        raise RuntimeError("can not select a valid tiling mode.")

    obj.global_init()

    switch = {1: obj.compute}

    switch[obj.mode]()
    return 0
