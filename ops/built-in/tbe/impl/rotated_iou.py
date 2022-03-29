# Copyright 2022 Huawei Technologies Co., Ltd
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
rotated_iou
"""

from topi.cce import util
import te.platform as tbe_platform
from te.utils import para_check
from te import tik
from impl.rotated_overlaps import Constant
from impl.rotated_overlaps import RotatedOverlaps


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
class RotatedIou(RotatedOverlaps):
    """
    The class for RotatedIou.
    """

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def __init__(self, boxes, query_boxes, iou, trans, mode, is_cross, v_threshold, e_threshold, kernel_name):
        """
        class init
        """
        RotatedOverlaps.__init__(self, boxes, query_boxes, iou, trans, kernel_name)
        self.v_threshold = v_threshold
        self.e_threshold = e_threshold

        self.area_of_boxes_ub = None

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def record_intersection_point_core(self):
        """
        record_intersection_point_core
        """
        self.BC_x.set_as(self.b2_x1 - self.b1_x2)
        self.BC_y.set_as(self.b2_y1 - self.b1_y2)
        self.BD_x.set_as(self.b2_x2 - self.b1_x2)
        self.BD_y.set_as(self.b2_y2 - self.b1_y2)
        self.AC_x.set_as(self.b2_x1 - self.b1_x1)
        self.AC_y.set_as(self.b2_y1 - self.b1_y1)
        self.AD_x.set_as(self.b2_x2 - self.b1_x1)
        self.AD_y.set_as(self.b2_y2 - self.b1_y1)
        # func: 'x = ((x1-x2) * (x3*y4-x4*y3) - (x3-x4) * (x1*y2-x2*y1)) / ((x3-x4) * (y1-y2) - (x1-x2)*(y3-y4))'
        # func: 'y = ((y1-y2) * (x3*y4-x4*y3) - (y3-y4) * (x1*y2-x2*y1)) / ((x3-x4) * (y1-y2) - (x1-x2)*(y3-y4))'
        self.direct_AC_AD.set_as(self.AC_x * self.AD_y - self.AC_y * self.AD_x)
        self.direct_BC_BD.set_as(self.BC_x * self.BD_y - self.BC_y * self.BD_x)
        with self.tik_instance.if_scope(self.direct_AC_AD * self.direct_BC_BD < self.e_threshold):
            self.direct_CA_CB.set_as(self.AC_x * self.BC_y - self.AC_y * self.BC_x)
            self.direct_DA_DB.set_as(self.AD_x * self.BD_y - self.AD_y * self.BD_x)
            with self.tik_instance.if_scope(self.direct_CA_CB * self.direct_DA_DB < self.e_threshold):
                self.b1_x1_x2.set_as(self.b1_x1 - self.b1_x2)
                self.b1_y1_y2.set_as(self.b1_y1 - self.b1_y2)
                self.b2_x1_x2.set_as(self.b2_x1 - self.b2_x2)
                self.b2_y1_y2.set_as(self.b2_y1 - self.b2_y2)
                self.tmp_1.set_as(self.b1_x1 * self.b1_y2 - self.b1_y1 * self.b1_x2)
                self.tmp_2.set_as(self.b2_x1 * self.b2_y2 - self.b2_y1 * self.b2_x2)

                self.denominator.set_as(self.b2_x1_x2 * self.b1_y1_y2 - self.b1_x1_x2 * self.b2_y1_y2)
                self.numerator_x.set_as(self.b1_x1_x2 * self.tmp_2 - self.tmp_1 * self.b2_x1_x2)
                self.numerator_y.set_as(self.b1_y1_y2 * self.tmp_2 - self.tmp_1 * self.b2_y1_y2)

                self.corners_ub[self.corners_num].set_as(self.numerator_x / self.denominator)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.numerator_y / self.denominator)
                self.corners_num.set_as(self.corners_num + 1)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def record_vertex_point(self, b2_idx):
        """
        record_vertex_point
        """
        self.corners_num.set_as(0)

        # func: b1 for input boxes & b2 for input query_boxes
        self.b1_x1.set_as(self.x1_of_boxes_ub[self.b1_offset])
        self.b1_x2.set_as(self.x2_of_boxes_ub[self.b1_offset])
        self.b1_x3.set_as(self.x3_of_boxes_ub[self.b1_offset])
        self.b1_x4.set_as(self.x4_of_boxes_ub[self.b1_offset])

        self.b1_y1.set_as(self.y1_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y2_of_boxes_ub[self.b1_offset])
        self.b1_y3.set_as(self.y3_of_boxes_ub[self.b1_offset])
        self.b1_y4.set_as(self.y4_of_boxes_ub[self.b1_offset])

        self.b2_x1.set_as(self.x1_of_boxes_ub[b2_idx])
        self.b2_x2.set_as(self.x2_of_boxes_ub[b2_idx])
        self.b2_x3.set_as(self.x3_of_boxes_ub[b2_idx])
        self.b2_x4.set_as(self.x4_of_boxes_ub[b2_idx])

        self.b2_y1.set_as(self.y1_of_boxes_ub[b2_idx])
        self.b2_y2.set_as(self.y2_of_boxes_ub[b2_idx])
        self.b2_y3.set_as(self.y3_of_boxes_ub[b2_idx])
        self.b2_y4.set_as(self.y4_of_boxes_ub[b2_idx])

        # check b1
        # func: 'AD = (x4-x1, y4-y1)'
        self.AD_x.set_as(self.b2_x4 - self.b2_x1)
        self.AD_y.set_as(self.b2_y4 - self.b2_y1)
        # func: 'AB = (x2-x1, y2-y1)'
        self.AB_x.set_as(self.b2_x2 - self.b2_x1)
        self.AB_y.set_as(self.b2_y2 - self.b2_y1)

        self.AB_AB.set_as(self.AB_x * self.AB_x + self.AB_y * self.AB_y)
        self.AD_AD.set_as(self.AD_x * self.AD_x + self.AD_y * self.AD_y)

        self.AP_x.set_as(self.b1_x1 - self.b2_x1)
        self.AP_y.set_as(self.b1_y1 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(
                tik.all(self.AP_AB >= self.v_threshold, self.AP_AB + self.v_threshold <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(
                    tik.all(self.AP_AD >= self.v_threshold, self.AP_AD + self.v_threshold <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b1_x1)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y1)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x2 - self.b2_x1)
        self.AP_y.set_as(self.b1_y2 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(
                tik.all(self.AP_AB >= self.v_threshold, self.AP_AB + self.v_threshold <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(
                    tik.all(self.AP_AD >= self.v_threshold, self.AP_AD + self.v_threshold <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b1_x2)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y2)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x3 - self.b2_x1)
        self.AP_y.set_as(self.b1_y3 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(
                tik.all(self.AP_AB >= self.v_threshold, self.AP_AB + self.v_threshold <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(
                    tik.all(self.AP_AD >= self.v_threshold, self.AP_AD + self.v_threshold <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b1_x3)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y3)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x4 - self.b2_x1)
        self.AP_y.set_as(self.b1_y4 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(
                tik.all(self.AP_AB >= self.v_threshold, self.AP_AB + self.v_threshold <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(
                    tik.all(self.AP_AD >= self.v_threshold, self.AP_AD + self.v_threshold <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b1_x4)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y4)
                self.corners_num.set_as(self.corners_num + 1)

        # check b2
        # func: 'AD = (x4-x1, y4-y1)'
        self.AD_x.set_as(self.b1_x4 - self.b1_x1)
        self.AD_y.set_as(self.b1_y4 - self.b1_y1)
        # func: 'AB = (x2-x1, y2-y1)'
        self.AB_x.set_as(self.b1_x2 - self.b1_x1)
        self.AB_y.set_as(self.b1_y2 - self.b1_y1)

        self.AB_AB.set_as(self.AB_x * self.AB_x + self.AB_y * self.AB_y)
        self.AD_AD.set_as(self.AD_x * self.AD_x + self.AD_y * self.AD_y)

        self.AP_x.set_as(self.b2_x1 - self.b1_x1)
        self.AP_y.set_as(self.b2_y1 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(
                tik.all(self.AP_AB >= self.v_threshold, self.AP_AB + self.v_threshold <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(
                    tik.all(self.AP_AD >= self.v_threshold, self.AP_AD + self.v_threshold <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b2_x1)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y1)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x2 - self.b1_x1)
        self.AP_y.set_as(self.b2_y2 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(
                tik.all(self.AP_AB >= self.v_threshold, self.AP_AB + self.v_threshold <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(
                    tik.all(self.AP_AD >= self.v_threshold, self.AP_AD + self.v_threshold <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b2_x2)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y2)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x3 - self.b1_x1)
        self.AP_y.set_as(self.b2_y3 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(
                tik.all(self.AP_AB >= self.v_threshold, self.AP_AB + self.v_threshold <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(
                    tik.all(self.AP_AD >= self.v_threshold, self.AP_AD + self.v_threshold <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b2_x3)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y3)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x4 - self.b1_x1)
        self.AP_y.set_as(self.b2_y4 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(
                tik.all(self.AP_AB >= self.v_threshold, self.AP_AB + self.v_threshold <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(
                    self.AP_AD >= self.v_threshold, self.AP_AD + self.v_threshold <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b2_x4)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y4)
                self.corners_num.set_as(self.corners_num + 1)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def compute_core(self, task_idx):
        """
        single task
        """
        self.data_init()
        self.area_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="area_of_boxes_ub",
                                                         scope=tik.scope_ubuf)
        b1_area = self.tik_instance.Scalar(self.dtype)
        b2_area = self.tik_instance.Scalar(self.dtype)
        overlap = self.tik_instance.Scalar(self.dtype)
        with self.tik_instance.for_range(0, Constant.BLOCK) as i:
            self.ori_idx_fp16_ub[i].set_as(self.idx_fp32)
            self.idx_fp32.set_as(self.idx_fp32 + 1)
        with self.tik_instance.for_range(0, self.batch) as current_batch:
            self.trans_boxes(task_idx, current_batch)
            self.valid_box_num.set_as(0)
            self.tik_instance.h_mul(self.area_of_boxes_ub, self.h_of_boxes_ub, self.w_of_boxes_ub)
            # record the valid query_boxes's num
            with self.tik_instance.for_range(0, self.k) as idx:
                self.w_value.set_as(self.w_of_boxes_ub[idx])
                self.h_value.set_as(self.h_of_boxes_ub[idx])
                with self.tik_instance.if_scope(self.w_value * self.h_value > 0):
                    self.valid_box_num.set_as(self.valid_box_num + 1)
            self.mov_repeats.set_as((self.valid_box_num + Constant.BLOCK - 1) // Constant.BLOCK)
            with self.tik_instance.for_range(0, self.b1_batch) as b1_idx:
                self.tik_instance.vec_dup(Constant.BLOCK, self.overlap_ub, 0, self.mov_repeats, 1)
                self.b1_offset.set_as(self.k_align - self.b1_batch + b1_idx)
                b1_area.set_as(self.area_of_boxes_ub[self.b1_offset])
                with self.tik_instance.for_range(0, self.valid_box_num) as b2_idx:
                    self.record_vertex_point(b2_idx)
                    self.record_intersection_point(b2_idx)
                    b2_area.set_as(self.area_of_boxes_ub[b2_idx])
                    with self.tik_instance.if_scope(self.corners_num == 3):
                        self.b1_x1.set_as(self.corners_ub[0])
                        self.b1_y1.set_as(self.corners_ub[Constant.BLOCK])
                        self.get_area_of_triangle(1, 2)
                        with self.tik_instance.if_scope(self.value > 0):
                            overlap.set_as(self.value / 2)
                        with self.tik_instance.else_scope():
                            overlap.set_as(-1 * self.value / 2)
                        with self.tik_instance.if_scope(b1_area + b2_area - overlap > 0):
                            self.overlap_ub[b2_idx].set_as(
                                overlap / (b1_area + b2_area - overlap + Constant.EPSILON))
                    with self.tik_instance.if_scope(self.corners_num > 3):
                        self.sum_area_of_triangles(b2_idx)
                        overlap.set_as(self.value / 2)
                        with self.tik_instance.if_scope(b1_area + b2_area - overlap > 0):
                            self.overlap_ub[b2_idx].set_as(
                                overlap / (b1_area + b2_area - overlap + Constant.EPSILON))
                self.tik_instance.data_move(
                    self.overlaps_gm[self.k * (task_idx * self.b1_batch + b1_idx + current_batch * self.n)],
                    self.overlap_ub, 0, 1, self.mov_repeats, 0, 0)


# 'pylint:disable=too-many-arguments, disable=too-many-statements
@tbe_platform.fusion_manager.fusion_manager.register("rotated_iou")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def rotated_iou(boxes, query_boxes, iou, trans=False, mode="iou", is_cross=True, v_threshold=0, e_threshold=0,
                kernel_name="rotated_iou"):
    """
    Function: compute the rotated boxes's iou.
    Modify : 2021-12-01

    Init base parameters
    Parameters
    ----------
    input(boxes): dict
        data of input
    input(query_boxes): dict
        data of input
    output(iou): dict
        data of output

    Attributes:
    trans : bool
        true for 'xyxyt', false for 'xywht'
    mode: string
        with the value range of ['iou', 'iof'], only support 'iou' now.
    is_cross: bool
        cross calculation when it is True, and one-to-one calculation when it is False.

    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = RotatedIou(boxes, query_boxes, iou, trans, mode, is_cross, v_threshold, e_threshold, kernel_name)

    return op_obj.compute()
