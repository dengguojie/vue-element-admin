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
from te import tik


UB_SIZE = 240 * 1024  # size of 310 ai core ub buffer
AI_CORE_NUM = 2


def _ceil_div_offline(value, factor):
    return (value + factor - 1) // factor


def _check_and_return_anchor_num(input0, input1, output):
    if (input0.get('dtype') != 'float16' or input1.get('dtype') != 'float16'
            or output.get('dtype') != 'float16'):
        raise RuntimeError("data type should be all float16")
    if input0.get('shape') == (1, 64, 38, 64, 16) and output.get('shape') == (
            100, 64, 14, 14, 16):
        return output.get('shape')[0]
    if input0.get('shape') == (1, 68, 40, 128, 16) and output.get('shape') == (
            300, 68, 17, 17, 16):
        return output.get('shape')[0]
    raise RuntimeError(
        "input should be either 1, 64, 38, 64, 16 with output 100, 64, 14, 14, 16/"
        "or (1, 68, 40, 128, 16) with output 100, 64, 14, 14, 16")


class ScopeCropAndResize:
    """
    Parameters
    ----------
    kernel_name : kernel name, default value is "cropandresize"
    function_description : extracts crops from the input image tensor and resizes them using bilinear \
                           sampling to a common output size
                           only supports input shape (1, 64, 38, 64, 16), crop_size (14, 14) \
                           or input shape (1, 68, 40, 128, 16), crop_size (17, 17)
    input0: dict shape dtype format of feature map
    input1: dict shape dtype format of map box
    output0: dict shape dtype format of output
    Returns
    -------
    None
    """

    def __init__(self, input0, intput1, output0, kernel_name="cropandresize"):
        self.kernel_name = kernel_name
        self.anchor_num = _check_and_return_anchor_num(input0, intput1, output0)
        self.fmp_shape = input0.get('shape')
        self.box_shape = intput1.get('shape')
        self.out_shape = output0.get('shape')
        self.input_n = self.fmp_shape[0]
        self.output_c = self.fmp_shape[1] * 16  # support align to 16
        self.roi_resize_w = self.out_shape[3]
        self.roi_resize_h = self.out_shape[2]
        self.input_w = self.fmp_shape[3]
        self.input_h = self.fmp_shape[2]
        self.output_w = self.out_shape[3]
        self.output_h = self.out_shape[2]
        self.align_w = _ceil_div_offline(self.output_w, 16) * 16
        self.align_h = _ceil_div_offline(self.output_h, 16) * 16
        self.aicore_num = AI_CORE_NUM

        self.tik_inst = tik.Tik(tik.Dprofile("v100", "mini"))
        self.featuremap = self.tik_inst.Tensor("float16", self.fmp_shape,
                                               name="featuremap_gm",
                                               scope=tik.scope_gm)
        self.mapboxout = self.tik_inst.Tensor("float16", self.box_shape,
                                              name="mapboxout_gm",
                                              scope=tik.scope_gm)
        self.cropandresize = self.tik_inst.Tensor("float16", self.out_shape,
                                                  name="cropandresize_gm",
                                                  scope=tik.scope_gm)

    def compute(self):
        last_core_index = self.aicore_num - 1
        batch_num_per_core = self.anchor_num // self.aicore_num
        with self.tik_inst.for_range(0, self.aicore_num,
                                     block_num=self.aicore_num) as i:
            block_offset = self.tik_inst.Scalar("int32", init_value=0)
            block_offset.set_as(i * batch_num_per_core)
            with self.tik_inst.if_scope(i != last_core_index):
                self._cropandresize_compute_each_core(block_offset,
                                                      batch_num_per_core)
            with self.tik_inst.else_scope():
                self._cropandresize_compute_each_core(block_offset,
                                                      batch_num_per_core)

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.featuremap, self.mapboxout],
                               outputs=[self.cropandresize])
        return self.tik_inst

    def _preprare_cropandresize_compute_buf(self):
        # ymin, xmin, ymax, xmax
        self.ymin_ub = self.tik_inst.Tensor("float16", (self.pad_anchor_num,), tik.scope_ubuf, "ymin_ub")
        self.xmin_ub = self.tik_inst.Tensor("float16", (self.pad_anchor_num,), tik.scope_ubuf, "xmin_ub")
        self.ymax_ub = self.tik_inst.Tensor("float16", (self.pad_anchor_num,), tik.scope_ubuf, "ymax_ub")
        self.xmax_ub = self.tik_inst.Tensor("float16", (self.pad_anchor_num,), tik.scope_ubuf, "xmax_ub")
        self.ymin_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,),
                                                 tik.scope_ubuf, "ymin_ub_fp32")
        self.xmin_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,),
                                                 tik.scope_ubuf, "xmin_ub_fp32")
        self.ymax_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,),
                                                 tik.scope_ubuf, "ymax_ub_fp32")
        self.xmax_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,),
                                                 tik.scope_ubuf, "xmax_ub_fp32")
        self.y1_ub = self.tik_inst.Tensor("float16", (self.pad_anchor_num,), tik.scope_ubuf, "y1_ub")
        self.x1_ub = self.tik_inst.Tensor("float16", (self.pad_anchor_num,), tik.scope_ubuf, "x1_ub")
        self.y2_ub = self.tik_inst.Tensor("float16", (self.pad_anchor_num,), tik.scope_ubuf, "y2_ub")
        self.x2_ub = self.tik_inst.Tensor("float16", (self.pad_anchor_num,), tik.scope_ubuf, "x2_ub")
        self.y1_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,), tik.scope_ubuf, "y1_ub_fp32")
        self.x1_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,), tik.scope_ubuf, "x1_ub_fp32")
        self.y2_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,), tik.scope_ubuf, "y2_ub_fp32")
        self.x2_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,), tik.scope_ubuf, "x2_ub_fp32")
        self.ymin_float_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,), tik.scope_ubuf,
                                                       "ymin_float_ub_fp32")
        self.xmin_float_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,), tik.scope_ubuf,
                                                       "xmin_float_ub_fp32")
        self.width_scale_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,), tik.scope_ubuf,
                                                        "width_scale_ub_fp32")
        self.height_scale_ub_fp32 = self.tik_inst.Tensor("float32", (self.pad_anchor_num,), tik.scope_ubuf,
                                                         "height_scale_ub_fp32")
        self.mapboxout_ub = self.tik_inst.Tensor("float16", self.box_shape, name="mapboxout_ub",
                                                 scope=tik.scope_ubuf)

    def _preprare_cropandresize_temp_buf(self):
        self.loss_ub_fp32 = self.tik_inst.Tensor("float32", (self.align_h,), tik.scope_ubuf, "loss_ub_fp32")
        self.dis_ub = self.tik_inst.Tensor("float16", (self.align_h,), tik.scope_ubuf, "dis_ub")
        self.dis_ub_fp32 = self.tik_inst.Tensor("float32", (self.align_h,), tik.scope_ubuf, "dis_ub_fp32")
        self.dis_index_ub = self.tik_inst.Tensor("int32", (self.align_h,), tik.scope_ubuf, "dis_index_ub")
        self.in_y_ub = self.tik_inst.Tensor("float16", (self.align_h,), tik.scope_ubuf, "in_y_ub")
        self.in_x_ub = self.tik_inst.Tensor("float16", (self.align_h,), tik.scope_ubuf, "in_x_ub")
        self.in_y_ub_fp32 = self.tik_inst.Tensor("float32", (self.align_h,), tik.scope_ubuf, "in_y_ub_fp32")
        self.in_x_ub_fp32 = self.tik_inst.Tensor("float32", (self.align_h,), tik.scope_ubuf, "in_x_ub_fp32")
        self.y_lerp_ub = self.tik_inst.Tensor("float16", (self.align_h,), tik.scope_ubuf, "y_lerp_ub")
        self.x_lerp_ub = self.tik_inst.Tensor("float16", (self.align_h,), tik.scope_ubuf, "x_lerp_ub")
        self.y_lerp_ub_fp32 = self.tik_inst.Tensor("float32", (self.align_h,), tik.scope_ubuf, "y_lerp_ub_fp32")
        self.x_lerp_ub_fp32 = self.tik_inst.Tensor("float32", (self.align_h,), tik.scope_ubuf, "x_lerp_ub_fp32")
        self.y_int_ub = self.tik_inst.Tensor("int32", (self.align_h,), tik.scope_ubuf, "y_int_ub")
        self.y_float_ub = self.tik_inst.Tensor("float16", (self.align_h,), tik.scope_ubuf, "y_float_ub")
        self.top_y_index_ub = self.tik_inst.Tensor("int32", (self.align_h,), tik.scope_ubuf, "top_y_index_ub")
        self.bottom_y_index_ub = self.tik_inst.Tensor("int32", (self.align_h,), tik.scope_ubuf,
                                                      "bottom_y_index_ub")
        self.left_x_index_ub = self.tik_inst.Tensor("int32", (self.align_h,), tik.scope_ubuf, "left_x_index_ub")
        self.right_x_index_ub = self.tik_inst.Tensor("int32", (self.align_h,), tik.scope_ubuf,
                                                     "right_x_index_ub")
        self.x_int_ub = self.tik_inst.Tensor("int32", (self.align_w,), tik.scope_ubuf, "x_int_ub")
        self.x_float_ub = self.tik_inst.Tensor("float16", (self.align_w,), tik.scope_ubuf, "x_float_ub")
        self.x_float_ub_fp32 = self.tik_inst.Tensor("float32", (self.align_w,),
                                                    tik.scope_ubuf, "x_float_ub_fp32")
        self.y_float_ub_fp32 = self.tik_inst.Tensor("float32", (self.align_w,),
                                                    tik.scope_ubuf, "y_float_ub_fp32")
        self.bot_right_ub = self.tik_inst.Tensor("float16", (self.output_c,),
                                                 tik.scope_ubuf, "bot_right_ub")
        self.bot_left_ub = self.tik_inst.Tensor("float16", (self.output_c,), tik.scope_ubuf, "bot_left_ub")
        self.top_right_ub = self.tik_inst.Tensor("float16", (self.output_c,), tik.scope_ubuf, "top_right_ub")
        self.top_left_ub = self.tik_inst.Tensor("float16", (self.output_c,), tik.scope_ubuf, "top_left_ub")

        self.temp_ub = self.tik_inst.Tensor("float16", (self.output_c,), tik.scope_ubuf, "temp_ub")
        self.result_ub = self.tik_inst.Tensor("float16", (self.output_c,), tik.scope_ubuf, "result_ub")
        self.top_ub = self.tik_inst.Tensor("float16", (self.output_c,), tik.scope_ubuf, "top_ub")
        self.bottom_ub = self.tik_inst.Tensor("float16", (self.output_c,), tik.scope_ubuf, "bottom_ub")

    def _preprare_cropandresize_temp_scalar(self):
        self.pad_anchor_num = _ceil_div_offline(self.anchor_num, 16) * 16

        self.height_scale_fp32 = self.tik_inst.Scalar("float32")
        self.width_scale_fp32 = self.tik_inst.Scalar("float32")

        self.div = self.tik_inst.Scalar("float32")
        self.x_lerp = self.tik_inst.Scalar("float16")
        self.y_lerp = self.tik_inst.Scalar("float16")
        self.y1_fp32 = self.tik_inst.Scalar("float32")
        self.x1_fp32 = self.tik_inst.Scalar("float32")

        self.top_y = self.tik_inst.Scalar("int32")
        self.bot_y = self.tik_inst.Scalar("int32")
        self.left_x = self.tik_inst.Scalar("int32")
        self.right_x = self.tik_inst.Scalar("int32")

        self.reswidth = self.tik_inst.Scalar("float32")
        self.resheight = self.tik_inst.Scalar("float32")
        self.reswidth.set_as(self.input_w - 1.0)
        self.resheight.set_as(self.input_h - 1.0)
        self.div.set_as(1.0 / (self.roi_resize_w - 1))
        self.left = self.pad_anchor_num % 64
        self.re_time = self.pad_anchor_num // 64
        self.left_index = self.pad_anchor_num - self.left

    def _calc_width_scale_repeat(self):
        if self.re_time > 0:
            self.tik_inst.vmuls(64, self.ymin_ub, self.ymin_ub, 0., self.re_time, 1, 1, 4, 4)
            self.tik_inst.vmuls(64, self.xmin_ub, self.xmin_ub, 0., self.re_time, 1, 1, 4, 4)
            self.tik_inst.vmuls(64, self.ymax_ub, self.ymax_ub, 0., self.re_time, 1, 1, 4, 4)
            self.tik_inst.vmuls(64, self.xmax_ub, self.xmax_ub, 0., self.re_time, 1, 1, 4, 4)

            # fp32
            self.tik_inst.vconv(64, "", self.y1_ub_fp32, self.y1_ub, self.re_time, 1, 1, 8, 4)
            self.tik_inst.vconv(64, "", self.x1_ub_fp32, self.x1_ub, self.re_time, 1, 1, 8, 4)
            self.tik_inst.vconv(64, "", self.y2_ub_fp32, self.y2_ub, self.re_time, 1, 1, 8, 4)
            self.tik_inst.vconv(64, "", self.x2_ub_fp32, self.x2_ub, self.re_time, 1, 1, 8, 4)

            self.tik_inst.vmuls(64, self.ymin_ub_fp32, self.y1_ub_fp32, self.resheight, self.re_time,
                                1, 1, 8, 8)
            self.tik_inst.vmuls(64, self.xmin_ub_fp32, self.x1_ub_fp32, self.reswidth, self.re_time,
                                1, 1, 8, 8)
            self.tik_inst.vmuls(64, self.ymax_ub_fp32, self.y2_ub_fp32, self.resheight, self.re_time,
                                1, 1, 8, 8)
            self.tik_inst.vmuls(64, self.xmax_ub_fp32, self.x2_ub_fp32, self.reswidth, self.re_time,
                                1, 1, 8, 8)

            self.tik_inst.data_move(self.ymin_float_ub_fp32, self.ymin_ub_fp32, 0, 1,
                                    (self.pad_anchor_num - self.left) // 8, 0, 0)
            self.tik_inst.data_move(self.xmin_float_ub_fp32, self.xmin_ub_fp32, 0, 1,
                                    (self.pad_anchor_num - self.left) // 8, 0, 0)

            # fp32
            self.tik_inst.vmuls(64, self.ymin_ub_fp32, self.ymin_ub_fp32, self.div, self.re_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(64, self.xmin_ub_fp32, self.xmin_ub_fp32, self.div, self.re_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(64, self.ymax_ub_fp32, self.ymax_ub_fp32, self.div, self.re_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(64, self.xmax_ub_fp32, self.xmax_ub_fp32, self.div, self.re_time, 1, 1, 8, 8)

            # fp32
            # height_scale equals to (y2 - y1) * (image_height - 1) / (crop_height - 1)
            self.tik_inst.vsub(64, self.height_scale_ub_fp32, self.ymax_ub_fp32, self.ymin_ub_fp32,
                               self.re_time, 1, 1, 1, 8, 8, 8)

            # width_scale equals to (x2 - x1) * (image_width - 1) / (crop_width - 1)
            self.tik_inst.vsub(64, self.width_scale_ub_fp32, self.xmax_ub_fp32, self.xmin_ub_fp32,
                               self.re_time, 1, 1, 1, 8, 8, 8)

    def _calc_width_scale_res(self):
        if self.left > 0:
            self.tik_inst.vmuls(self.left, self.ymin_ub[self.left_index], self.ymin_ub[self.left_index], 0.,
                                1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.xmin_ub[self.left_index], self.xmin_ub[self.left_index], 0.,
                                1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.ymax_ub[self.left_index], self.ymax_ub[self.left_index], 0.,
                                1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.xmax_ub[self.left_index], self.xmax_ub[self.left_index], 0.,
                                1, 1, 1, 8, 8)
            self.tik_inst.vconv(self.left, "", self.y1_ub_fp32[self.left_index],
                                self.y1_ub[self.left_index], 1, 1, 1, 8, 4)
            self.tik_inst.vconv(self.left, "", self.x1_ub_fp32[self.left_index], self.x1_ub[self.left_index],
                                1, 1, 1, 8, 4)
            self.tik_inst.vconv(self.left, "", self.y2_ub_fp32[self.left_index],
                                self.y2_ub[self.left_index], 1, 1, 1, 8, 4)
            self.tik_inst.vconv(self.left, "", self.x2_ub_fp32[self.left_index],
                                self.x2_ub[self.left_index], 1, 1, 1, 8, 4)
            self.tik_inst.vmuls(self.left, self.ymin_ub_fp32[self.left_index], self.y1_ub_fp32[self.left_index],
                                self.resheight, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.xmin_ub_fp32[self.left_index], self.x1_ub_fp32[self.left_index],
                                self.reswidth, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.ymax_ub_fp32[self.left_index], self.y2_ub_fp32[self.left_index],
                                self.resheight, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.xmax_ub_fp32[self.left_index], self.x2_ub_fp32[self.left_index],
                                self.reswidth, 1, 1, 1, 8, 8)
            self.tik_inst.data_move(self.ymin_float_ub_fp32[self.left_index],
                                    self.ymin_ub_fp32[self.left_index], 0, 1, self.left // 8, 0, 0)
            self.tik_inst.data_move(self.xmin_float_ub_fp32[self.left_index],
                                    self.xmin_ub_fp32[self.left_index], 0, 1, self.left // 8, 0, 0)
            # 1/16.0
            self.tik_inst.vmuls(self.left, self.ymin_ub_fp32[self.left_index], self.ymin_ub_fp32[self.left_index],
                                self.div, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.xmin_ub_fp32[self.left_index], self.xmin_ub_fp32[self.left_index],
                                self.div, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.ymax_ub_fp32[self.left_index], self.ymax_ub_fp32[self.left_index],
                                self.div, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.left, self.xmax_ub_fp32[self.left_index], self.xmax_ub_fp32[self.left_index],
                                self.div, 1, 1, 1, 8, 8)
            # height_scale equals to (y2 - y1) * (image_height - 1) / (crop_height - 1)
            self.tik_inst.vsub(self.left, self.height_scale_ub_fp32[self.left_index],
                               self.ymax_ub_fp32[self.left_index],
                               self.ymin_ub_fp32[self.left_index], 1, 1, 1, 1, 8, 8, 8)
            # width_scale equals to (x2 - x1) * (image_width - 1) / (crop_width - 1)
            self.tik_inst.vsub(self.left, self.width_scale_ub_fp32[self.left_index],
                               self.xmax_ub_fp32[self.left_index],
                               self.xmin_ub_fp32[self.left_index], 1, 1, 1, 1, 8, 8, 8)

    def _calc_y_lerp(self):
        self.tik_inst.vconv(self.output_h, "", self.y_float_ub, self.y_int_ub, 1, 1, 1, 4, 8, 1.0)
        self.tik_inst.vconv(self.output_h, "", self.y_float_ub_fp32, self.y_float_ub, 1, 1, 1, 8, 4)  # in_x
        self.tik_inst.vmuls(self.output_h, self.y_float_ub_fp32, self.y_float_ub_fp32, self.height_scale_fp32,
                            1, 1, 1, 8, 8)

        # in_x equals to x1 * (image_width - 1) + x * width_scale
        self.tik_inst.vadds(self.output_h, self.in_y_ub_fp32, self.y_float_ub_fp32, self.y1_fp32, 1, 1, 1,
                            8, 8)

        self.tik_inst.vconv(self.output_h, "", self.in_y_ub, self.in_y_ub_fp32, 1, 1, 1, 4, 8)
        self.tik_inst.vconv(self.output_h, "floor", self.top_y_index_ub, self.in_y_ub, 1, 1, 1, 8, 4)
        self.tik_inst.vconv(self.output_h, "", self.in_y_ub, self.top_y_index_ub, 1, 1, 1, 4, 8, 1.0)

        self.tik_inst.vconv(self.output_h, "", self.loss_ub_fp32, self.in_y_ub, 1, 1, 1, 8, 4)  # turn 16 to 32
        self.tik_inst.vsub(self.output_h, self.dis_ub_fp32, self.in_y_ub_fp32, self.loss_ub_fp32,
                           1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vconv(self.output_h, "", self.dis_ub, self.dis_ub_fp32, 1, 1, 1, 4, 8)

        self.tik_inst.vconv(self.output_h, "ceil", self.dis_index_ub, self.dis_ub, 1, 1, 1, 8, 4)  # ceil
        self.tik_inst.vadd(self.output_h, self.bottom_y_index_ub, self.top_y_index_ub, self.dis_index_ub, 1,
                           1, 1, 1, 8, 8, 8)

        self.tik_inst.vconv(self.output_h, "floor", self.dis_index_ub, self.dis_ub, 1, 1, 1, 8, 4)

        self.tik_inst.vadd(self.output_h, self.top_y_index_ub, self.top_y_index_ub, self.dis_index_ub, 1,
                           1, 1, 1, 8, 8, 8)

        self.tik_inst.vconv(self.output_h, "", self.y_float_ub, self.top_y_index_ub, 1, 1, 1, 4, 8, 1.0)
        self.tik_inst.vconv(self.output_h, "", self.y_float_ub_fp32, self.y_float_ub, 1, 1, 1, 8, 4)
        self.tik_inst.vsub(self.output_h, self.y_lerp_ub_fp32, self.in_y_ub_fp32, self.y_float_ub_fp32, 1,
                           1, 1, 1, 8, 8, 8)  # x_lerp equals to in_x - left_x_index
        self.tik_inst.vconv(self.output_h, "", self.y_lerp_ub, self.y_lerp_ub_fp32, 1, 1, 1, 4, 8)

    def _calc_x_lerp(self):
        self.tik_inst.vconv(self.output_w, "", self.x_float_ub, self.x_int_ub, 1, 1, 1, 4, 8, 1.0)
        self.tik_inst.vconv(self.output_w, "", self.x_float_ub_fp32, self.x_float_ub, 1, 1, 1, 8, 4)  # in_x
        self.tik_inst.vmuls(self.output_w, self.x_float_ub_fp32, self.x_float_ub_fp32, self.width_scale_fp32,
                            1, 1, 1, 8, 8)

        self.tik_inst.vadds(self.output_w, self.in_x_ub_fp32, self.x_float_ub_fp32, self.x1_fp32, 1, 1, 1,
                            8, 8)  # in_x equals to x1 * (image_width - 1) + x * width_scale

        self.tik_inst.vconv(self.output_w, "", self.in_x_ub, self.in_x_ub_fp32, 1, 1, 1, 4, 8)
        self.tik_inst.vconv(self.output_w, "floor", self.left_x_index_ub, self.in_x_ub, 1, 1, 1, 8, 4)
        self.tik_inst.vconv(self.output_w, "", self.in_x_ub, self.left_x_index_ub, 1, 1, 1, 4, 8, 1.0)

        self.tik_inst.vconv(self.output_w, "", self.loss_ub_fp32, self.in_x_ub, 1, 1, 1, 8, 4)
        self.tik_inst.vsub(self.output_w, self.dis_ub_fp32, self.in_x_ub_fp32, self.loss_ub_fp32,
                           1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vconv(self.output_w, "", self.dis_ub, self.dis_ub_fp32, 1, 1, 1, 4, 8)

        self.tik_inst.vconv(self.output_w, "ceil", self.dis_index_ub, self.dis_ub, 1, 1, 1, 8, 4)  # ceil
        self.tik_inst.vadd(self.output_w, self.right_x_index_ub, self.left_x_index_ub, self.dis_index_ub, 1,
                           1, 1, 1, 8, 8, 8)

        self.tik_inst.vconv(self.output_w, "floor", self.dis_index_ub, self.dis_ub, 1, 1, 1, 8, 4)

        self.tik_inst.vadd(self.output_w, self.left_x_index_ub, self.left_x_index_ub, self.dis_index_ub, 1,
                           1, 1, 1, 8, 8, 8)

        self.tik_inst.vconv(self.output_w, "", self.x_float_ub, self.left_x_index_ub, 1, 1, 1, 4, 8,
                            1.0)  # x_lerp
        self.tik_inst.vconv(self.output_w, "", self.x_float_ub_fp32, self.x_float_ub, 1, 1, 1, 8, 4)
        self.tik_inst.vsub(self.output_w, self.x_lerp_ub_fp32, self.in_x_ub_fp32, self.x_float_ub_fp32, 1,
                           1, 1, 1, 8, 8, 8)  # x_lerp equals to in_x - left_x_index
        self.tik_inst.vconv(self.output_w, "", self.x_lerp_ub, self.x_lerp_ub_fp32, 1, 1, 1, 4, 8)

    def _main_compute_branch_1(self, times, h_out):
        with self.tik_inst.for_range(0, self.output_w) as w_out:
            self.tik_inst.vector_dup(128, self.result_ub, 0, self.reapeat128, 1, 8)
            if (self.input_n * self.output_c) % 128 > 0:
                self.tik_inst.vector_dup((self.input_n * self.output_c) % 128,
                                         self.result_ub[self.input_n * self.output_c -
                                                        (self.input_n * self.output_c) % 128],
                                         0, 1, 1, 8)
            self.tik_inst.data_move(self.cropandresize[times, 0, h_out, w_out, 0], self.result_ub,
                                    0, 1, self.reapeat16, 0, self.output_h * self.output_w - 1)

    def _main_compute_branch_2_left(self):
        self.left_data = (self.input_n * self.output_c) % 128
        self.left_data_index = self.input_n * self.output_c - self.left_data
        if self.left_data > 0:
            # top equals to top_left + (top_right - top_left) * x_lerp
            self.tik_inst.vsub(self.left_data, self.temp_ub[self.left_data_index],
                               self.top_right_ub[self.left_data_index],
                               self.top_left_ub[self.left_data_index], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(self.left_data, self.temp_ub[self.left_data_index],
                                self.temp_ub[self.left_data_index], self.x_lerp, 1, 1, 1, 8, 8)
            self.tik_inst.vadd(self.left_data, self.top_ub[self.left_data_index],
                               self.top_left_ub[self.left_data_index],
                               self.temp_ub[self.left_data_index], 1, 1, 1, 1, 8, 8, 8)

            # bottom equals to bottom_left + (bottom_right - bottom_left) * x_lerp
            self.tik_inst.vsub(self.left_data, self.temp_ub[self.left_data_index],
                               self.bot_right_ub[self.left_data_index],
                               self.bot_left_ub[self.left_data_index], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(self.left_data, self.temp_ub[self.left_data_index],
                                self.temp_ub[self.left_data_index], self.x_lerp, 1, 1, 1, 8, 8)
            self.tik_inst.vadd(self.left_data, self.bottom_ub[self.left_data_index],
                               self.bot_left_ub[self.left_data_index],
                               self.temp_ub[self.left_data_index], 1, 1, 1, 1, 8, 8, 8)

            # output equals to top + (bottom - top) * y_lerp
            self.tik_inst.vsub(self.left_data, self.temp_ub[self.left_data_index],
                               self.bottom_ub[self.left_data_index],
                               self.top_ub[self.left_data_index], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(self.left_data, self.temp_ub[self.left_data_index],
                                self.temp_ub[self.left_data_index], self.y_lerp, 1, 1, 1, 8, 8)
            self.tik_inst.vadd(self.left_data, self.result_ub[self.left_data_index],
                               self.top_ub[self.left_data_index],
                               self.temp_ub[self.left_data_index], 1, 1, 1, 1, 8, 8, 8)

    def _main_compute_branch_2(self, times, h_out):
        with self.tik_inst.for_range(0, self.output_w) as w_out:
            self.left_x.set_as(self.left_x_index_ub[w_out])
            self.right_x.set_as(self.right_x_index_ub[w_out])
            self.x_lerp.set_as(self.x_lerp_ub[w_out])

            with self.tik_inst.if_scope(tik.any(self.right_x > (self.input_w - 1), self.left_x < 0)):
                self.tik_inst.vector_dup(128, self.result_ub, 0, self.reapeat128, 1, 8)
                if (self.input_n * self.output_c) % 128 > 0:
                    self.tik_inst.vector_dup((self.input_n * self.output_c) % 128,
                                             self.result_ub[self.input_n * self.output_c -
                                                            (self.input_n * self.output_c) % 128],
                                             0, 1, 1, 8)
                self.tik_inst.data_move(self.cropandresize[times, 0, h_out, w_out, 0], self.result_ub,
                                        0, 1, self.reapeat16, 0, self.output_h * self.output_w - 1)

            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.top_left_ub, self.featuremap[0, 0, self.top_y, self.left_x, 0],
                                        0, self.reapeat16, 1, self.input_w * self.input_h - 1, 0)
                self.tik_inst.data_move(self.top_right_ub, self.featuremap[0, 0, self.top_y, self.right_x, 0],
                                        0, self.reapeat16, 1, self.input_w * self.input_h - 1, 0)
                self.tik_inst.data_move(self.bot_left_ub, self.featuremap[0, 0, self.bot_y, self.left_x, 0], 0,
                                        self.reapeat16, 1, self.input_w * self.input_h - 1, 0)
                self.tik_inst.data_move(self.bot_right_ub, self.featuremap[0, 0, self.bot_y, self.right_x, 0], 0,
                                        self.reapeat16, 1, self.input_w * self.input_h - 1, 0)

                # top equals to top_left + (top_right - top_left) * x_lerp
                self.tik_inst.vsub(128, self.temp_ub, self.top_right_ub, self.top_left_ub,
                                   self.reapeat128, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vmuls(128, self.temp_ub, self.temp_ub, self.x_lerp, self.reapeat128, 1, 1, 8, 8)
                self.tik_inst.vadd(128, self.top_ub, self.top_left_ub, self.temp_ub, self.reapeat128, 1, 1, 1, 8, 8, 8)

                # bottom equals to bottom_left + (bottom_right - bottom_left) * x_lerp
                self.tik_inst.vsub(128, self.temp_ub, self.bot_right_ub, self.bot_left_ub, self.reapeat128,
                                   1, 1, 1, 8, 8, 8)
                self.tik_inst.vmuls(128, self.temp_ub, self.temp_ub, self.x_lerp, self.reapeat128, 1, 1, 8, 8)
                self.tik_inst.vadd(128, self.bottom_ub, self.bot_left_ub, self.temp_ub, self.reapeat128,
                                   1, 1, 1, 8, 8, 8)

                # output equals to top + (bottom - top) * y_lerp
                self.tik_inst.vsub(128, self.temp_ub, self.bottom_ub, self.top_ub, self.reapeat128, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vmuls(128, self.temp_ub, self.temp_ub, self.y_lerp, self.reapeat128, 1, 1, 8, 8)
                self.tik_inst.vadd(128, self.result_ub, self.top_ub, self.temp_ub, self.reapeat128, 1, 1, 1, 8, 8, 8)
                self._main_compute_branch_2_left()

                self.tik_inst.data_move(self.cropandresize[times, 0, h_out, w_out, 0],
                                        self.result_ub, 0, self.reapeat16, 1, 0, self.output_h * self.output_w - 1)

    def _cropandresize_compute_each_core(self, batch_offset, batch_num):
        self._preprare_cropandresize_temp_scalar()
        self._preprare_cropandresize_compute_buf()
        self._preprare_cropandresize_temp_buf()

        self.tik_inst.data_move(self.mapboxout_ub, self.mapboxout, 0, 1, self.box_shape[0] * 4 // 16, 0, 0)

        with self.tik_inst.for_range(0, self.box_shape[0]) as i:
            self.y1_ub[i] = self.mapboxout_ub[i, 0]
            self.x1_ub[i] = self.mapboxout_ub[i, 1]
            self.y2_ub[i] = self.mapboxout_ub[i, 2]
            self.x2_ub[i] = self.mapboxout_ub[i, 3]

        self._calc_width_scale_repeat()
        self._calc_width_scale_res()

        # only support self.output_h & self.output_w < 64
        with self.tik_inst.for_range(0, self.output_h) as h_out:
            self.y_int_ub[h_out] = h_out

        with self.tik_inst.for_range(0, self.output_w) as w_out:
            self.x_int_ub[w_out] = w_out

        self.reapeat16 = ((self.input_n * self.output_c) // 16)
        self.reapeat128 = ((self.input_n * self.output_c) // 128)

        with self.tik_inst.for_range(batch_offset, batch_offset + batch_num) as times:
            self.height_scale_fp32.set_as(self.height_scale_ub_fp32[times])
            self.width_scale_fp32.set_as(self.width_scale_ub_fp32[times])
            self.y1_fp32.set_as(self.ymin_float_ub_fp32[times])
            self.x1_fp32.set_as(self.xmin_float_ub_fp32[times])
            # calc y_lerp
            self._calc_y_lerp()

            # calc x_lerp
            self._calc_x_lerp()

            with self.tik_inst.for_range(0, self.output_h) as h_out:
                self.top_y.set_as(self.top_y_index_ub[h_out])
                self.bot_y.set_as(self.bottom_y_index_ub[h_out])
                self.y_lerp.set_as(self.y_lerp_ub[h_out])

                with self.tik_inst.if_scope(tik.any(self.bot_y > (self.input_h - 1), self.top_y < 0)):
                    self._main_compute_branch_1(times, h_out)
                with self.tik_inst.else_scope():
                    self._main_compute_branch_2(times, h_out)


def fasterrcnn_cropandresize_tik(input0, input1, output0, kernel_name="cropandresize"):
    """
    Extract crops from the input image tensor and resizes them using bilinear sampling to a common output size.
    This op supports input shape are (1, 64, 38, 64, 16), crop_size is (14, 14)
                                     or (1, 68, 40, 128, 16), crop_size is (17, 17)
    """
    obj = ScopeCropAndResize(input0, input1, output0, kernel_name)
    obj.compute()
