# -*- coding:utf-8 -*-
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

SCALE_FACTOR_0 = 10.0
SCALE_FACTOR_1 = 10.0
SCALE_FACTOR_2 = 5.0
SCALE_FACTOR_3 = 5.0


# input1: y_min, x_min, y_max, x_max
# input2: ty, tx, th, tw->anchor
# output: ymin, xmin, ymax, xmax

class FirstStageDecode():
    def __init__(self, tik_instance):
        self.tik_inst = tik_instance

    def _basic_buff_apply(self, proposal_box):
        self.y_min_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="y_min_fp32", scope=tik.scope_ubuf)
        self.x_min_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="x_min_fp32", scope=tik.scope_ubuf)
        self.y_max_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="y_max_fp32", scope=tik.scope_ubuf)
        self.x_max_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="x_max_fp32", scope=tik.scope_ubuf)
        self.ty_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="ty_fp32", scope=tik.scope_ubuf)
        self.tx_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="tx_fp32", scope=tik.scope_ubuf)
        self.th_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="th_fp32", scope=tik.scope_ubuf)
        self.tw_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="tw_fp32", scope=tik.scope_ubuf)

        self.width_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="width_ub", scope=tik.scope_ubuf)
        self.height_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="height_ub", scope=tik.scope_ubuf)
        self.width_div2 = self.tik_inst.Tensor("float32", (proposal_box,), name="width_div2", scope=tik.scope_ubuf)
        self.height_div2 = self.tik_inst.Tensor("float32", (proposal_box,), name="height_div2",
                                                scope=tik.scope_ubuf)

        self.ycenter_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="ycenter_ub", scope=tik.scope_ubuf)
        self.xcenter_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="xcenter_ub", scope=tik.scope_ubuf)
        self.w_div2_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="w_div2_ub", scope=tik.scope_ubuf)
        self.h_div2_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="h_div2_ub", scope=tik.scope_ubuf)

        self.tynew = self.tik_inst.Tensor("float32", (proposal_box,), name="tynew", scope=tik.scope_ubuf)
        self.txnew = self.tik_inst.Tensor("float32", (proposal_box,), name="txnew", scope=tik.scope_ubuf)
        self.thnew = self.tik_inst.Tensor("float32", (proposal_box,), name="thnew", scope=tik.scope_ubuf)
        self.twnew = self.tik_inst.Tensor("float32", (proposal_box,), name="twnew", scope=tik.scope_ubuf)

        self.thnew_fp16 = self.tik_inst.Tensor("float16", (proposal_box,), name="thnew_fp16", scope=tik.scope_ubuf)
        self.twnew_fp16 = self.tik_inst.Tensor("float16", (proposal_box,), name="twnew_fp16", scope=tik.scope_ubuf)

        self.width_exp_ub = self.tik_inst.Tensor("float16", (proposal_box,), name="width_exp_ub",
                                                 scope=tik.scope_ubuf)
        self.height_exp_ub = self.tik_inst.Tensor("float16", (proposal_box,), name="height_exp_ub",
                                                  scope=tik.scope_ubuf)

        self.width_exp_ub_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="width_exp_ub_fp32",
                                                      scope=tik.scope_ubuf)
        self.height_exp_ub_fp32 = self.tik_inst.Tensor("float32", (proposal_box,), name="height_exp_ub_fp32",
                                                       scope=tik.scope_ubuf)

        self.w_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="w_ub", scope=tik.scope_ubuf)
        self.h_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="h_ub", scope=tik.scope_ubuf)

        self.ty_ha_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="ty_ha_ub", scope=tik.scope_ubuf)
        self.tx_wa_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="tx_wa_ub", scope=tik.scope_ubuf)
        self.ycenter_de_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="w_ub", scope=tik.scope_ubuf)
        self.xcenter_de_ub = self.tik_inst.Tensor("float32", (proposal_box,), name="h_ub", scope=tik.scope_ubuf)
        self.num_div2 = self.tik_inst.Scalar("float32")
        self.index = self.tik_inst.Scalar("int32")
        self.factor0_div_scalar = self.tik_inst.Scalar("float32")
        self.factor1_div_scalar = self.tik_inst.Scalar("float32")
        self.factor2_div_scalar = self.tik_inst.Scalar("float32")
        self.factor3_div_scalar = self.tik_inst.Scalar("float32")

    def decode_process(self, y_min, x_min, y_max, x_max, t_y, t_x, t_h, t_w):
        proposal_box = 64
        self._basic_buff_apply(proposal_box)
        shape_1 = y_min.shape[0]
        till = shape_1 // proposal_box
        leftdata_index = till * proposal_box
        leftdata_mask = y_min.shape[0] - leftdata_index
        self.num_div2.set_as(0.5)
        self.factor0_div_scalar.set_as(1. / SCALE_FACTOR_0)
        self.factor1_div_scalar.set_as(1. / SCALE_FACTOR_1)
        self.factor2_div_scalar.set_as(1. / SCALE_FACTOR_2)
        self.factor3_div_scalar.set_as(1. / SCALE_FACTOR_3)
        with self.tik_inst.for_range(0, till) as till_count:
            self.index.set_as(till_count * proposal_box)
            repeat = 1
            self._decode_compute(y_min, x_min, y_max, x_max, t_y, t_x, t_h, t_w, repeat, 64)
        if leftdata_mask:
            self.index.set_as(till * proposal_box)
            self._decode_compute(y_min, x_min, y_max, x_max, t_y, t_x, t_h, t_w, 1, leftdata_mask)

    def _decode_compute(self, y_min, x_min, y_max, x_max, t_y, t_x, t_h, t_w, repeat, mask):
        # compute anchor w,h ,x,y
        # wa equals to xmax - xmin
        # ha equals to ymax - ymin
        # ycenter_a equals to ymin + height / 2.
        # xcenter_a equals to xmin + width / 2.
        self.tik_inst.vconv(mask, '', self.y_min_fp32, y_min[self.index], repeat, 1, 1, 8, 4)
        self.tik_inst.vconv(mask, '', self.x_min_fp32, x_min[self.index], repeat, 1, 1, 8, 4)
        self.tik_inst.vconv(mask, '', self.y_max_fp32, y_max[self.index], repeat, 1, 1, 8, 4)
        self.tik_inst.vconv(mask, '', self.x_max_fp32, x_max[self.index], repeat, 1, 1, 8, 4)
        self.tik_inst.vconv(mask, '', self.ty_fp32, t_y[self.index], repeat, 1, 1, 8, 4)
        self.tik_inst.vconv(mask, '', self.tx_fp32, t_x[self.index], repeat, 1, 1, 8, 4)
        self.tik_inst.vconv(mask, '', self.tw_fp32, t_w[self.index], repeat, 1, 1, 8, 4)
        self.tik_inst.vconv(mask, '', self.th_fp32, t_h[self.index], repeat, 1, 1, 8, 4)

        self.tik_inst.vsub(mask, self.width_ub[0], self.x_max_fp32[0], self.x_min_fp32[0], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(mask, self.height_ub[0], self.y_max_fp32[0], self.y_min_fp32[0], repeat,
                           1, 1, 1, 8, 8, 8)
        self.tik_inst.vmuls(mask, self.width_div2[0], self.width_ub[0], self.num_div2, repeat, 1, 1, 8, 8, 0)
        self.tik_inst.vmuls(mask, self.height_div2[0], self.height_ub[0], self.num_div2, repeat, 1, 1, 8, 8, 0)
        self.tik_inst.vadd(mask, self.ycenter_ub[0], self.y_min_fp32[0], self.height_div2[0], repeat,
                           1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, self.xcenter_ub[0], self.x_min_fp32[0], self.width_div2[0], repeat,
                           1, 1, 1, 8, 8, 8)

        # ty equals to ty/10, scalar[0]
        # tx equals to tx/10, scalar[1]
        # th equals to th/5, scalar[2]
        # tw equals to tw/5, scalar[3]
        self.tik_inst.vmuls(mask, self.tynew[0], self.ty_fp32[0], self.factor0_div_scalar, repeat, 1, 1, 8, 8, 0)
        self.tik_inst.vmuls(mask, self.txnew[0], self.tx_fp32[0], self.factor1_div_scalar, repeat, 1, 1, 8, 8, 0)
        self.tik_inst.vmuls(mask, self.thnew[0], self.th_fp32[0], self.factor2_div_scalar, repeat, 1, 1, 8, 8, 0)
        self.tik_inst.vmuls(mask, self.twnew[0], self.tw_fp32[0], self.factor3_div_scalar, repeat, 1, 1, 8, 8, 0)

        #  ycenter_a, xcenter_a, ha, wa equals to self.ycenter_ub, self.xcenter_ub, self.height_ub, self.width_ub
        #  w equals to tf.exp(tw) * wa
        #  h equals to tf.exp(th) * ha
        # ty, tx, th, tw equals to tf.unstack(tf.transpose(rel_codes))

        self.tik_inst.vconv(mask, '', self.twnew_fp16, self.twnew, repeat, 1, 1, 4, 8)
        self.tik_inst.vconv(mask, '', self.thnew_fp16, self.thnew, repeat, 1, 1, 4, 8)

        self.tik_inst.vexp(mask, self.width_exp_ub[0], self.twnew_fp16[0], repeat, 1, 1, 8, 8, 0)
        self.tik_inst.vexp(mask, self.height_exp_ub[0], self.thnew_fp16[0], repeat, 1, 1, 8, 8, 0)

        self.tik_inst.vconv(mask, '', self.width_exp_ub_fp32, self.width_exp_ub, repeat, 1, 1, 8, 4)
        self.tik_inst.vconv(mask, '', self.height_exp_ub_fp32, self.height_exp_ub, repeat, 1, 1, 8, 4)

        self.tik_inst.vmul(mask, self.w_ub[0], self.width_exp_ub_fp32[0], self.width_ub[0], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(mask, self.h_ub[0], self.height_exp_ub_fp32[0], self.height_ub[0], repeat, 1, 1, 1, 8, 8, 8)

        # ycenter equals to ty * ha + ycenter_a
        # xcenter equals to tx * wa + xcenter_a
        self.tik_inst.vmul(mask, self.ty_ha_ub[0], self.tynew[0], self.height_ub[0], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(mask, self.tx_wa_ub[0], self.txnew[0], self.width_ub[0], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, self.ycenter_de_ub[0], self.ty_ha_ub[0], self.ycenter_ub[0], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, self.xcenter_de_ub[0], self.tx_wa_ub[0], self.xcenter_ub[0], repeat, 1, 1, 1, 8, 8, 8)

        # ymin equals to ycenter - h / 2.
        # xmin equals to xcenter - w / 2.
        # ymax equals to ycenter + h / 2.
        # xmax equals to xcenter + w / 2.
        self.tik_inst.vmuls(mask, self.w_div2_ub[0], self.w_ub[0], self.num_div2, repeat, 1, 1, 8, 8, 0)
        self.tik_inst.vmuls(mask, self.h_div2_ub[0], self.h_ub[0], self.num_div2, repeat, 1, 1, 8, 8, 0)
        self.tik_inst.vsub(mask, self.y_min_fp32[0], self.ycenter_de_ub[0], self.h_div2_ub[0], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(mask, self.x_min_fp32[0], self.xcenter_de_ub[0], self.w_div2_ub[0], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, self.y_max_fp32[0], self.ycenter_de_ub[0], self.h_div2_ub[0], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, self.x_max_fp32[0], self.xcenter_de_ub[0], self.w_div2_ub[0], repeat, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vconv(mask, '', y_min[self.index], self.y_min_fp32, repeat, 1, 1, 4, 8)
        self.tik_inst.vconv(mask, '', x_min[self.index], self.x_min_fp32, repeat, 1, 1, 4, 8)
        self.tik_inst.vconv(mask, '', y_max[self.index], self.y_max_fp32, repeat, 1, 1, 4, 8)
        self.tik_inst.vconv(mask, '', x_max[self.index], self.x_max_fp32, repeat, 1, 1, 4, 8)


def decode(tik_instance, y_min, x_min, y_max, x_max, t_y, t_x, t_h, t_w):
    """
    function: The function of deocde is to calculate the final detection frame position and size by calculating the
              proposal_boxes and refined_box_encodings
    @param [in/out]: y_min, x_min, y_max, x_max: coordinate of proposal_boxes
    @param [in]: t_y, t_x, t_h, t_w: coordinate of refined_box_encodings
    """
    decode_obj = FirstStageDecode(tik_instance)
    decode_obj.decode_process(y_min, x_min, y_max, x_max, t_y, t_x, t_h, t_w)
