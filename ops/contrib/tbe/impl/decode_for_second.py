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


class Decode:
    """
    @param [in] tik_ins
    # proposal_box
    @param [in] y_min: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] x_min: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] y_max: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] x_max: shape(NUM_CLASS, PROPOSAL_BOX)
    # refine_Boxlist
    @param [in] refine_y: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] refine_x: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] refine_h: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] refine_w: shape(NUM_CLASS, PROPOSAL_BOX)
    --------
    output: refine_x, refine_h, refine_w
    """

    def __init__(self, tik_instance, y_min, x_min, y_max, x_max, refine_y, refine_x, refine_h, refine_w, case):
        self.tik_instance = tik_instance
        self.y_min = y_min
        self.x_min = x_min
        self.y_max = y_max
        self.x_max = x_max
        self.refine_y = refine_y
        self.refine_x = refine_x
        self.refine_h = refine_h
        self.refine_w = refine_w
        self.case = case

    def decode_process(self):
        if self.case == 0:
            till = 6  # set till value according to UB size
            num_class = self.y_min.shape[0] // till

        else:
            till = 1
            num_class = self.y_min.shape[0] // till
        proposal_box = self.y_min.shape[1]
        shape_1 = proposal_box * num_class
        repeat = shape_1 // 128
        left_data_index = repeat * 128
        left_data_mask = shape_1 - left_data_index  # left mask

        # index of left mask, start index of left data in axis of num_class
        offset_dim1 = num_class - (left_data_mask // proposal_box) - 1
        # start index of left data in axis of proposal_box
        offset_dim0 = proposal_box - (left_data_mask % proposal_box)

        width_ub = self.tik_instance.Tensor("float16", (num_class, proposal_box), name="width_ub",
                                            scope=tik.scope_ubuf)
        height_ub = self.tik_instance.Tensor("float16", (num_class, proposal_box), name="height_ub",
                                             scope=tik.scope_ubuf)
        ycenter_ub = self.tik_instance.Tensor("float16", (num_class, proposal_box), name="ycenter_ub",
                                              scope=tik.scope_ubuf)
        xcenter_ub = self.tik_instance.Tensor("float16", (num_class, proposal_box), name="xcenter_ub",
                                              scope=tik.scope_ubuf)
        self.num_div2 = self.tik_instance.Scalar("float16")
        self.num_div2.set_as(0.5)

        with self.tik_instance.for_range(0, till) as till_count:
            config_list = [num_class, till_count, 128, repeat, 0, 0]
            self.decode_compute(width_ub, height_ub, ycenter_ub, xcenter_ub, config_list)
            if (left_data_mask > 0):
                config_list = [num_class, till_count, left_data_mask, 1, offset_dim1, offset_dim0]
                self.decode_compute(width_ub, height_ub, ycenter_ub, xcenter_ub, config_list)
            self.tik_instance.data_move(self.refine_y[num_class * till_count, 0],
                                        self.y_min[num_class * till_count, 0],
                                        0, 1, num_class * proposal_box // 16, 8, 8)
            self.tik_instance.data_move(self.refine_x[num_class * till_count, 0],
                                        self.x_min[num_class * till_count, 0],
                                        0, 1, num_class * proposal_box // 16, 8, 8)
            self.tik_instance.data_move(self.refine_h[num_class * till_count, 0],
                                        self.y_max[num_class * till_count, 0],
                                        0, 1, num_class * proposal_box // 16, 8, 8)
            self.tik_instance.data_move(self.refine_w[num_class * till_count, 0],
                                        self.x_max[num_class * till_count, 0],
                                        0, 1, num_class * proposal_box // 16, 8, 8)

    def _decode_compute_xy_center(self, tik_ins, width, height, xcenter, ycenter, num_cls, till, mask, repeat,
                                  off_dim1, off_dim0):
        tik_ins.vsub(mask, width[off_dim1, off_dim0], self.x_max[num_cls * till + off_dim1, off_dim0],
                     self.x_min[num_cls * till + off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vsub(mask, height[off_dim1, off_dim0], self.y_max[num_cls * till + off_dim1, off_dim0],
                     self.y_min[num_cls * till + off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vmuls(mask, xcenter[off_dim1, off_dim0], width[off_dim1, off_dim0], self.num_div2,
                      repeat, 1, 1, 8, 8, 0)
        tik_ins.vmuls(mask, ycenter[off_dim1, off_dim0], height[off_dim1, off_dim0], self.num_div2,
                      repeat, 1, 1, 8, 8, 0)
        tik_ins.vadd(mask, ycenter[off_dim1, off_dim0], self.y_min[num_cls * till + off_dim1, off_dim0],
                     ycenter[off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vadd(mask, xcenter[off_dim1, off_dim0], self.x_min[num_cls * till + off_dim1, off_dim0],
                     xcenter[off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)

    def decode_compute(self, width, height, ycenter, xcenter, config_list):
        num_cls, till, mask, repeat, off_dim1, off_dim0 = config_list
        tik_ins = self.tik_instance
        # compute anchor w,h ,x,y
        # width is defined as:  xmax - xmin
        # height is defined as: ymax - ymin
        # ycenter is defined as: ymin + height / 2.
        # xcenter is defined as: xmin + width / 2.
        self._decode_compute_xy_center(tik_ins, width, height, xcenter, ycenter, num_cls, till, mask, repeat,
                                       off_dim1, off_dim0)

        # scalar[0] is used to compute refine_y, which is refine_y /= 10;
        # scalar[1] is used to compute refine_x, which is refine_x /= 10;
        # scalar[2] is used to compute refine_h, which is refine_h /= 5;
        # scalar[3] is used to compute refine_w, which is refine_w /= 5;
        tik_ins.vmuls(mask, self.refine_y[num_cls * till + off_dim1, off_dim0],
                      self.refine_y[num_cls * till + off_dim1, off_dim0], 1 / SCALE_FACTOR_0, repeat, 1, 1, 8, 8, 0)
        tik_ins.vmuls(mask, self.refine_x[num_cls * till + off_dim1, off_dim0],
                      self.refine_x[num_cls * till + off_dim1, off_dim0], 1 / SCALE_FACTOR_1, repeat, 1, 1, 8, 8, 0)
        tik_ins.vmuls(mask, self.refine_h[num_cls * till + off_dim1, off_dim0],
                      self.refine_h[num_cls * till + off_dim1, off_dim0], 1 / SCALE_FACTOR_2, repeat, 1, 1, 8, 8, 0)
        tik_ins.vmuls(mask, self.refine_w[num_cls * till + off_dim1, off_dim0],
                      self.refine_w[num_cls * till + off_dim1, off_dim0], 1 / SCALE_FACTOR_3, repeat, 1, 1, 8, 8, 0)

        # ycenter_a, xcenter_a, ha, wa is corresponds to ycenter_ub, xcenter_ub, height_ub, width_ub
        # w is defined as: exp(refine_w) * wa
        # h is defined as: exp(refine_h) * ha
        tik_ins.vexp(mask, self.refine_w[num_cls * till + off_dim1, off_dim0],
                     self.refine_w[num_cls * till + off_dim1, off_dim0], repeat, 1, 1, 8, 8, 0)
        tik_ins.vexp(mask, self.refine_h[num_cls * till + off_dim1, off_dim0],
                     self.refine_h[num_cls * till + off_dim1, off_dim0], repeat, 1, 1, 8, 8, 0)
        tik_ins.vmul(mask, self.refine_w[num_cls * till + off_dim1, off_dim0], self.refine_w[num_cls * till + off_dim1,
                                                                                             off_dim0],
                     width[off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vmul(mask, self.refine_h[num_cls * till + off_dim1, off_dim0], self.refine_h[num_cls * till + off_dim1,
                                                                                             off_dim0],
                     height[off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)

        # ycenter is defined as: refine_y * ha + ycenter_a
        # xcenter is defined as: refine_x * wa + xcenter_a
        tik_ins.vmul(mask, self.refine_y[num_cls * till + off_dim1, off_dim0], self.refine_y[num_cls * till + off_dim1,
                                                                                             off_dim0],
                     height[off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vmul(mask, self.refine_x[num_cls * till + off_dim1, off_dim0], self.refine_x[num_cls * till + off_dim1,
                                                                                             off_dim0],
                     width[off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vadd(mask, ycenter[off_dim1, off_dim0], self.refine_y[num_cls * till + off_dim1, off_dim0],
                     ycenter[off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vadd(mask, xcenter[off_dim1, off_dim0], self.refine_x[num_cls * till + off_dim1, off_dim0],
                     xcenter[off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)

        # ymin is defined as: ycenter - h / 2.
        # xmin is defined as: xcenter - w / 2.
        # ymax is defined as: ycenter + h / 2.
        # xmax is defined as: xcenter + w / 2.
        tik_ins.vmuls(mask, self.refine_w[num_cls * till + off_dim1, off_dim0],
                      self.refine_w[num_cls * till + off_dim1, off_dim0], self.num_div2, repeat, 1, 1, 8, 8, 0)
        tik_ins.vmuls(mask, self.refine_h[num_cls * till + off_dim1, off_dim0],
                      self.refine_h[num_cls * till + off_dim1, off_dim0], self.num_div2, repeat, 1, 1, 8, 8, 0)
        tik_ins.vsub(mask, self.y_min[num_cls * till + off_dim1, off_dim0], ycenter[off_dim1, off_dim0],
                     self.refine_h[num_cls * till + off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vsub(mask, self.x_min[num_cls * till + off_dim1, off_dim0], xcenter[off_dim1, off_dim0],
                     self.refine_w[num_cls * till + off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vadd(mask, self.y_max[num_cls * till + off_dim1, off_dim0], ycenter[off_dim1, off_dim0],
                     self.refine_h[num_cls * till + off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)
        tik_ins.vadd(mask, self.x_max[num_cls * till + off_dim1, off_dim0], xcenter[off_dim1, off_dim0],
                     self.refine_w[num_cls * till + off_dim1, off_dim0], repeat, 1, 1, 1, 8, 8, 8)


def decode(tik_instance, y_min, x_min, y_max, x_max, refine_y, refine_x, refine_h, refine_w, case):
    """
    @param [in] tik_ins
    # proposal_box
    @param [in] y_min: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] x_min: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] y_max: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] x_max: shape(NUM_CLASS, PROPOSAL_BOX)
    # refine_Boxlist
    @param [in] refine_y: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] refine_x: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] refine_h: shape(NUM_CLASS, PROPOSAL_BOX)
    @param [in] refine_w: shape(NUM_CLASS, PROPOSAL_BOX)
    --------
    output: refine_x, refine_h, refine_w
    """

    decode_ancher = Decode(tik_instance, y_min, x_min, y_max, x_max, refine_y, refine_x, refine_h, refine_w, case)
    decode_ancher.decode_process()
