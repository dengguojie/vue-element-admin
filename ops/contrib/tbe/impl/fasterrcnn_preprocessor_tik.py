# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
from . import get_version

tik, TBE_VERSION = get_version.get_tbe_version()


def _gcd(val_a, val_b):
    if val_a < val_b:
        val_a, val_b = val_b, val_a
    while val_b != 0:
        val_a, val_b = val_b, val_a % val_b
    return val_a


class ScopePreprocessor:
    """
    Parameters
    ----------
    kernel_name : kernel name, default value is "preprocessor"
    function_description : preprocess input image
    input0 : dict shape dtype format of image
    output0 : dict shape dtype format of processed image
    -------
    Returns: None
    """

    def __init__(self, input0, output0, kernel_name="preprocessor"):
        self.kernel_name = kernel_name
        input_shape = input0.get("shape")
        output_shape = output0.get("shape")
        if input0.get('format') != 'NHWC' or output0.get('format') != 'NHWC':
            raise RuntimeError("input0 output0 format should be NHWC")

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.input_n = input_shape[0]
        self.input_h = input_shape[1]
        self.input_width = input_shape[2]
        self.input_channel = input_shape[3]
        self.output_h = output_shape[1]
        self.output_w = output_shape[2]
        self.input_num = self.input_n * _gcd(self.input_h, self.output_h)
        self.input_height = self.input_h // _gcd(self.input_h, self.output_h)
        self.output_height = self.output_h // _gcd(self.input_h, self.output_h)

        self.output_width = self.output_w // _gcd(self.input_width, self.output_w)

        self.height_scale = float(self.input_height) / self.output_height
        self.width_scale = float(self.input_width) / self.output_w
        self.data_a_gm = self.tik_instance.Tensor("float16", (
            self.input_num, self.input_height, self.input_width, self.input_channel), tik.scope_gm,
                                                  "data_a_gm")

        self.dst_gm = self.tik_instance.Tensor("float16", (
            1 + self.input_num, self.output_height, self.output_w, self.input_channel),
                                               tik.scope_gm, "dst_gm")

    def _preprocessor_alloc_buf(self):
        self.hin_int_ub = self.tik_instance.Tensor("int32", (self.output_height,), tik.scope_ubuf,
                                                   "hin_int_ub")
        self.hin_float_ub = self.tik_instance.Tensor("float16", (self.output_height,),
                                                     tik.scope_ubuf, "hin_float_ub")
        self.h_bottom_ub = self.tik_instance.Tensor("float16", (self.output_height,),
                                                    tik.scope_ubuf, "h_bottom_ub")
        self.h_top_ub = self.tik_instance.Tensor("float16", (self.output_height,), tik.scope_ubuf,
                                                 "h_top_ub")

        self.win_int_ub = self.tik_instance.Tensor("int32", (self.output_w,), tik.scope_ubuf,
                                                   "win_int_ub")
        self.win_float_ub = self.tik_instance.Tensor("float16", (self.output_w,), tik.scope_ubuf,
                                                     "win_float_ub")
        self.w_right_ub = self.tik_instance.Tensor("float16", (self.output_w,), tik.scope_ubuf,
                                                   "w_right_ub")
        self.w_left_ub = self.tik_instance.Tensor("float16", (self.output_w,), tik.scope_ubuf,
                                                  "w_left_ub")

        self.vector_locatevalue_temp = self.tik_instance.Tensor("float16", (self.input_num * 16,),
                                                                tik.scope_ubuf,
                                                                "vector_locatevalue_temp")
        self.vector_average = self.tik_instance.Tensor("float16", (self.input_num * 16,),
                                                       tik.scope_ubuf, "vector_average")

    def _preprocessor_calc_wint(self):
        with self.tik_instance.for_range(0, self.output_height) as h_outt:
            self.hin_int_ub[h_outt] = h_outt
        self.tik_instance.vconv(self.output_height, "", self.hin_float_ub, self.hin_int_ub, 1, 1, 1, 4, 8, 1.0)

        self.tik_instance.vmuls(self.output_height, self.hin_float_ub, self.hin_float_ub,
                                self.height_scale, 1, 1, 1, 8, 8, 0)

        self.tik_instance.vconv(self.output_height, "floor", self.hin_int_ub, self.hin_float_ub, 1, 1, 1, 8, 4)

        self.tik_instance.vconv(self.output_height, "", self.h_bottom_ub, self.hin_int_ub, 1, 1, 1, 4, 8, 1.0)

        self.tik_instance.vsub(self.output_height, self.h_bottom_ub, self.hin_float_ub,
                               self.h_bottom_ub, 1, 1, 1, 1, 8, 8, 8, 0)

        self.tik_instance.vadds(self.output_height, self.h_top_ub, self.h_bottom_ub, -1.0, 1, 1, 1, 8, 8, 0)

        self.tik_instance.vabs(self.output_height, self.h_top_ub, self.h_top_ub, 1, 1, 1, 8, 8, 0)

        with self.tik_instance.for_range(0, self.output_w) as w_ou:
            self.win_int_ub[w_ou] = w_ou
        self.tik_instance.vconv(64, "", self.win_float_ub, self.win_int_ub, self.output_w // 64, 1, 1, 4, 8, 1.0)

        self.tik_instance.vmuls(128, self.win_float_ub, self.win_float_ub, self.width_scale,
                                self.output_w // 128, 1, 1, 8, 8, 0)

        self.tik_instance.vconv(64, "floor", self.win_int_ub, self.win_float_ub,
                                self.output_w // 64, 1, 1, 8, 4)

        self.tik_instance.vconv(64, "", self.w_right_ub, self.win_int_ub, self.output_w // 64, 1, 1, 4, 8, 1.0)

        self.tik_instance.vsub(128, self.w_right_ub, self.win_float_ub, self.w_right_ub,
                               self.output_w // 128, 1, 1, 1, 8, 8, 8, 0)

        self.tik_instance.vadds(128, self.w_left_ub, self.w_right_ub, -1.0, self.output_w // 128, 1, 1, 8, 8, 0)

        self.tik_instance.vabs(128, self.w_left_ub, self.w_left_ub, self.output_w // 128, 1, 1, 8, 8, 0)

        self.tik_instance.vconv(64, "", self.win_float_ub, self.win_int_ub, self.output_w // 64, 1, 1, 4, 8, 1.0)

        with self.tik_instance.for_range(1, 64) as iter_t:
            self.tik_instance.vadds(16, self.win_float_ub[16 * iter_t],
                                    self.win_float_ub[(iter_t - 1) * 16], 15.0, 1, 1, 1, 8, 8, 0)
        self.tik_instance.vconv(64, "floor", self.win_int_ub, self.win_float_ub, 16, 1, 1, 8, 4)

    def _preprocessor_calc_h_cylce_lerp(self, h_out):
        self.h_top_value.set_as(self.h_top_ub[h_out])
        self.h_bottom_value.set_as(self.h_bottom_ub[h_out])
        self.tik_instance.vmuls(self.output_width, self.lerp_tr, self.w_right_ub, self.h_top_value,
                                1, 1, 1, 8, 8, 0)

        self.tik_instance.vmuls(self.output_width, self.lerp_br, self.w_right_ub,
                                self.h_bottom_value, 1, 1, 1, 8, 8, 0)

        self.tik_instance.vmuls(self.output_width, self.lerp_tl, self.w_left_ub, self.h_top_value,
                                1, 1, 1, 8, 8, 0)

        self.tik_instance.vmuls(self.output_width, self.lerp_bl, self.w_left_ub,
                                self.h_bottom_value, 1, 1, 1, 8, 8, 0)

    def _preprocessor_calc_w_cylce_locate(self, w_out, h_out):
        self.w_in.set_as(self.win_int_ub[w_out])
        self.tik_instance.data_move(self.vector_locatevalue_br, self.data_a_gm[
            0, self.h_in + self.h1p, self.w_in + self.w1p, 0], 0, self.input_num, 1, int(
            (self.input_channel * self.input_width * self.input_height // 16) - 1), 0)
        self.tik_instance.data_move(self.vector_locatevalue_bl,
                                    self.data_a_gm[0, self.h_in + self.h1p, self.w_in, 0], 0,
                                    self.input_num, 1, int((self.input_channel * self.input_width
                                                            * self.input_height // 16) - 1), 0)
        self.tik_instance.data_move(self.vector_locatevalue_tr,
                                    self.data_a_gm[0, self.h_in, self.w_in + self.w1p, 0], 0,
                                    self.input_num, 1, int((self.input_channel * self.input_width
                                                            * self.input_height // 16) - 1), 0)
        self.tik_instance.data_move(self.vector_locatevalue_tl,
                                    self.data_a_gm[0, self.h_in, self.w_in, 0], 0, self.input_num,
                                    1, int((self.input_channel * self.input_width * self.input_height // 16) - 1), 0)
        with self.tik_instance.if_scope(h_out == self.output_height - 1):
            self.tik_instance.data_move(self.vector_locatevalue_br[304], self.data_a_gm[
                self.input_num - 1, 14, self.w_in + self.w1p, 0], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.vector_locatevalue_bl[304],
                                        self.data_a_gm[self.input_num - 1, 14, self.w_in, 0], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.vector_locatevalue_tr[304], self.data_a_gm[
                self.input_num - 1, 14, self.w_in + self.w1p, 0], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.vector_locatevalue_tl[304],
                                        self.data_a_gm[self.input_num - 1, 14, self.w_in, 0], 0, 1, 1, 0, 0)

        self.num_br.set_as(self.lerp_br[w_out % self.output_width])
        self.num_bl.set_as(self.lerp_bl[w_out % self.output_width])
        self.num_tr.set_as(self.lerp_tr[w_out % self.output_width])
        self.num_tl.set_as(self.lerp_tl[w_out % self.output_width])

        self.tik_instance.vmuls(128, self.vector_locatevalue_br, self.vector_locatevalue_br,
                                self.num_br, 16 * self.input_num // 128, 1, 1, 8, 8, 0)

        self.tik_instance.vmuls(128, self.vector_locatevalue_bl, self.vector_locatevalue_bl,
                                self.num_bl, 16 * self.input_num // 128, 1, 1, 8, 8, 0)

        self.tik_instance.vmuls(128, self.vector_locatevalue_tr, self.vector_locatevalue_tr,
                                self.num_tr, 16 * self.input_num // 128, 1, 1, 8, 8, 0)

        self.tik_instance.vmuls(128, self.vector_locatevalue_tl, self.vector_locatevalue_tl,
                                self.num_tl, 16 * self.input_num // 128, 1, 1, 8, 8, 0)

        self.tik_instance.vadd(128, self.vector_locatevalue_br, self.vector_locatevalue_br,
                               self.vector_locatevalue_bl, 16 * self.input_num // 128, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(128, self.vector_locatevalue_br, self.vector_locatevalue_br,
                               self.vector_locatevalue_tr, 16 * self.input_num // 128, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(128, self.vector_locatevalue_br, self.vector_locatevalue_br,
                               self.vector_locatevalue_tl, 16 * self.input_num // 128, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(128, self.vector_locatevalue_br, self.vector_locatevalue_br,
                                0.00784313771874, 16 * self.input_num // 128, 1, 1, 8, 8)

        self.tik_instance.vadd(128, self.vector_locatevalue_br, self.vector_locatevalue_br,
                               self.vector_average, 16 * self.input_num // 128, 1, 1, 1, 8, 8, 8)

    def _preprocessor_calc_w_cylce_res_locate(self):
        res_data = (16 * self.input_num) % 128
        res_data_index = 16 * self.input_num // 128 * 128
        if res_data > 0:
            self.tik_instance.vmuls(res_data, self.vector_locatevalue_br[res_data_index],
                                    self.vector_locatevalue_br[res_data_index], self.num_br, 1, 1,
                                    1, 8, 8, 0)

            self.tik_instance.vmuls(res_data, self.vector_locatevalue_bl[res_data_index],
                                    self.vector_locatevalue_bl[res_data_index], self.num_bl, 1, 1,
                                    1, 8, 8, 0)

            self.tik_instance.vmuls(res_data, self.vector_locatevalue_tr[res_data_index],
                                    self.vector_locatevalue_tr[res_data_index], self.num_tr, 1, 1,
                                    1, 8, 8, 0)

            self.tik_instance.vmuls(res_data, self.vector_locatevalue_tl[res_data_index],
                                    self.vector_locatevalue_tl[res_data_index], self.num_tl, 1, 1,
                                    1, 8, 8, 0)

            self.tik_instance.vadd(res_data, self.vector_locatevalue_br[res_data_index],
                                   self.vector_locatevalue_br[res_data_index],
                                   self.vector_locatevalue_bl[res_data_index], 1, 1, 1, 1, 8, 8, 8)

            self.tik_instance.vadd(res_data, self.vector_locatevalue_br[res_data_index],
                                   self.vector_locatevalue_br[res_data_index],
                                   self.vector_locatevalue_tr[res_data_index], 1, 1, 1, 1, 8, 8, 8)

            self.tik_instance.vadd(res_data, self.vector_locatevalue_br[res_data_index],
                                   self.vector_locatevalue_br[res_data_index],
                                   self.vector_locatevalue_tl[res_data_index], 1, 1, 1, 1, 8, 8, 8)

            self.tik_instance.vmuls(res_data, self.vector_locatevalue_br[res_data_index],
                                    self.vector_locatevalue_br[res_data_index], 0.00784313771874, 1,
                                    1, 1, 8, 8)

            self.tik_instance.vadd(res_data, self.vector_locatevalue_br[res_data_index],
                                   self.vector_locatevalue_br[res_data_index],
                                   self.vector_average[res_data_index], 1, 1, 1, 1, 8, 8, 8)

    def _preprocessor_alloc_scalar(self):
        self.first = self.tik_instance.Scalar("float16")
        self.first.set_as(-1.0)
        self.second = self.tik_instance.Scalar("float16")
        self.second.set_as(-1.0)
        self.third = self.tik_instance.Scalar("float16")
        self.third.set_as(-1.0)
        with self.tik_instance.for_range(0, self.input_num) as times:
            self.vector_average[times * 16].set_as(self.first)
            self.vector_average[times * 16 + 1].set_as(self.second)
            self.vector_average[times * 16 + 2].set_as(self.third)
        self.num_br = self.tik_instance.Scalar("float16")
        self.num_bl = self.tik_instance.Scalar("float16")
        self.num_tr = self.tik_instance.Scalar("float16")
        self.num_tl = self.tik_instance.Scalar("float16")
        self.h_in = self.tik_instance.Scalar("int32")
        self.h1p = self.tik_instance.Scalar("int32")
        self.h1p.set_as(1)
        self.h_top_value = self.tik_instance.Scalar("float16")
        self.h_bottom_value = self.tik_instance.Scalar("float16")
        self.w_in = self.tik_instance.Scalar("int32")
        self.w1p = self.tik_instance.Scalar("int32")
        self.w1p.set_as(1)

    def preprocessor_compute(self):
        self._preprocessor_alloc_buf()
        self._preprocessor_calc_wint()
        self._preprocessor_alloc_scalar()

        with self.tik_instance.for_range(0, self.output_height) as h_out:
            self.lerp_br = self.tik_instance.Tensor("float16", (self.output_width,), tik.scope_ubuf, "lerp_br")
            self.lerp_tr = self.tik_instance.Tensor("float16", (self.output_width,), tik.scope_ubuf, "lerp_tr")
            self.lerp_tl = self.tik_instance.Tensor("float16", (self.output_width,), tik.scope_ubuf, "lerp_tl")
            self.lerp_bl = self.tik_instance.Tensor("float16", (self.output_width,), tik.scope_ubuf, "lerp_bl")
            self.h_in.set_as(self.hin_int_ub[h_out])

            self._preprocessor_calc_h_cylce_lerp(h_out)

            with self.tik_instance.if_scope(h_out == 2):
                self.tik_instance.data_move(self.vector_locatevalue_temp, self.dst_gm[0, 0, 0, 0],
                                            0, self.input_num, 1, int(
                        (self.input_channel * self.output_w * self.output_height // 16) - 1), 0)
            with self.tik_instance.for_range(0, self.output_w) as w_out:
                with self.tik_instance.if_scope(w_out == self.output_w - 1):
                    self.w1p.set_as(0)
                with self.tik_instance.else_scope():
                    self.w1p.set_as(1)
                self.vector_locatevalue_br = self.tik_instance.Tensor("float16",
                                                                      (self.input_num * 16,),
                                                                      tik.scope_ubuf,
                                                                      "vector_locatevalue_br")
                self.vector_locatevalue_bl = self.tik_instance.Tensor("float16",
                                                                      (self.input_num * 16,),
                                                                      tik.scope_ubuf,
                                                                      "vector_locatevalue_bl")
                self.vector_locatevalue_tr = self.tik_instance.Tensor("float16",
                                                                      (self.input_num * 16,),
                                                                      tik.scope_ubuf,
                                                                      "vector_locatevalue_tr")
                self.vector_locatevalue_tl = self.tik_instance.Tensor("float16",
                                                                      (self.input_num * 16,),
                                                                      tik.scope_ubuf,
                                                                      "vector_locatevalue_tl")

                self._preprocessor_calc_w_cylce_locate(w_out, h_out)
                self._preprocessor_calc_w_cylce_res_locate()

                self.tik_instance.data_move(
                    self.dst_gm[0, h_out, w_out, 0], self.vector_locatevalue_br, 0, self.input_num, 1, 0,
                    int((self.input_channel * self.output_w * self.output_height // 16) - 1))

        self.tik_instance.data_move(self.dst_gm[0, 0, 0, 0], self.vector_locatevalue_temp, 0, self.input_num, 1, 0,
                                    int((self.input_channel * self.output_w * self.output_height // 16) - 1))

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.data_a_gm],
                                   outputs=[self.dst_gm], enable_l2=False)
        return self.tik_instance


def fasterrcnn_preprocessor_tik(input0, output0, kernel_name="preprocessor"):
    """
    Preprocess the input image data, including resizing the image, data normalization,
    and subtracting the average value
    """
    obj = ScopePreprocessor(input0, output0, kernel_name)
    obj.preprocessor_compute()
