# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
iou_3d
"""

from topi.cce import util
import te.platform as tbe_platform
from te.utils import para_check
from te import tik


class Constant(object):
    """
    The class for constant.
    """
    # min val in float16
    MIN_VAL = -65504
    # float32 data'nums in 32B
    BLOCK = 8
    # float16 data'nums in 32B
    BATCH = 16
    # idx tag for {x, y, z, w, h, d, theta}
    X_IDX = 0
    Y_IDX = 1
    Z_IDX = 2
    W_IDX = 3
    H_IDX = 4
    D_IDX = 5
    T_IDX = 6
    # nums of box info
    INFOS = 7
    # val's idx in proposal
    VAL_IDX = 4
    # limit of k's size of query_boxes
    K_LIMIT = 2000
    # to avoid denominator zero
    EPSILON = 1e-6
    UNIT = 10
    HALF_UNIT = 5


# 'pylint: disable=locally-disabled, unused-argument, invalid-name
class Iou3D:
    """
    The class for Iou3D.
    """

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def __init__(self, boxes, query_boxes, iou, kernel_name):
        """
        class init
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        check_res = Iou3D.paras_check(boxes, query_boxes, iou, kernel_name)
        self.batch, self.n, self.k, self.dtype = check_res[0], check_res[1], check_res[2], check_res[3]
        self.kernel_name, self.task_num = kernel_name, self.n
        while self.task_num % 2 == 0 and self.task_num >= 64:
            self.task_num //= 2
        self.b1_batch = self.n // self.task_num
        if self.b1_batch >= Constant.BLOCK:
            self.b1_repeats = (self.b1_batch + Constant.BLOCK - 1) // Constant.BLOCK
        else:
            self.task_num, self.b1_batch, self.b1_repeats = self.n, 1, 1
        self.k_align = (self.k + Constant.BLOCK - 1 + self.b1_batch) // Constant.BLOCK * Constant.BLOCK
        self.repeats = self.k_align // Constant.BLOCK
        self.available_aicore_num = tik.Dprofile().get_aicore_num()
        self.used_aicore_num = self.available_aicore_num if self.task_num > self.available_aicore_num \
            else self.task_num
        self.batch_num_per_aicore = self.task_num // self.used_aicore_num
        self.batch_tail = self.task_num % self.used_aicore_num
        self.boxes_gm = self.tik_instance.Tensor(self.dtype, [self.batch, Constant.INFOS, self.n], name="boxes_gm",
                                                 scope=tik.scope_gm)
        self.query_boxes_gm = self.tik_instance.Tensor(self.dtype, [self.batch, Constant.INFOS, self.k],
                                                       name="query_boxes_gm", scope=tik.scope_gm)
        self.iou_gm = self.tik_instance.Tensor(self.dtype, [self.batch, self.n, self.k], name="iou_gm",
                                                    scope=tik.scope_gm, is_atomic_add=True)
        lis = [None] * Constant.UNIT
        self.idx_fp16_ub, self.ori_idx_fp16_ub, self.box_ub, self.overlap_ub, self.corners_ub, self.val_fp16_ub,\
            self.idx_int32_ub, self.proposal_ub, self.x_of_boxes_ub, self.y_of_boxes_ub = lis
        self.w_of_boxes_ub, self.h_of_boxes_ub, self.half_w_of_boxes_ub, self.half_h_of_boxes_ub, self.t_of_boxes_ub,\
            self.radian_t_of_boxes_ub, self.cos_t_of_boxes_ub, self.sin_t_of_boxes_ub, self.half_w_cos_of_boxes_ub,\
            self.half_w_sin_of_boxes_ub = lis
        self.half_h_cos_of_boxes_ub, self.half_h_sin_of_boxes_ub, self.x_sub_w_of_boxes_ub, self.y_sub_w_of_boxes_ub,\
            self.x_add_w_of_boxes_ub, self.y_add_w_of_boxes_ub, self.x1_of_boxes_ub, self.x2_of_boxes_ub,\
            self.x3_of_boxes_ub, self.x4_of_boxes_ub = lis
        self.y1_of_boxes_ub, self.y2_of_boxes_ub, self.y3_of_boxes_ub, self.y4_of_boxes_ub, self.x_tensor_ub,\
            self.y_tensor_ub, self.slope_tensor_ub, self.add_tensor_ub, self.abs_tensor_ub, self.tmp_tensor_ub = lis
        self.d_of_boxes_ub, self.work_tensor_ub, self.clockwise_idx_int32_ub, self.idx_fp32, self.min_val,\
            self.half, self.radian, self.value, self.w_value, self.h_value = lis
        self.valid_box_num, self.mov_repeats, self.corners_num, self.idx_right, self.idx_left, self.b1_offset,\
            self.b1_x, self.b1_y, self.b2_x, self.b2_y = lis
        self.b1_x1, self.b1_y1, self.b2_x1, self.b2_y1, self.b1_x2, self.b1_y2, self.b2_x2,\
            self.b2_y2, self.b1_x3, self.b1_y3 = lis
        self.b2_x3, self.b2_y3, self.b1_x4, self.b1_y4, self.b2_x4, self.b2_y4, self.AB_x, self.AB_y,\
            self.AC_x, self.AC_y = lis
        self.AD_x, self.AD_y, self.AP_x, self.AP_y, self.BC_x, self.BC_y, self.BD_x,\
            self.BD_y, self.AB_AB, self.AD_AD = lis
        self.AP_AB, self.AP_AD, self.direct_AC_AD, self.direct_BC_BD, self.direct_CA_CB,\
            self.direct_DA_DB, self.tmp_1, self.tmp_2, self.b1_x1_x2, self.b1_y1_y2 = lis
        self.b2_x1_x2, self.b2_y1_y2, self.denominator, self.numerator_x, self.numerator_y, self.half_d_of_boxes_ub,\
            self.z_sub_d_boxes_ub, self.z_add_d_boxes_ub, self.volume_of_boxes_ub, self.d_value = lis
        self.z_of_boxes_ub, self.max_of_min, self.min_of_max, self.real_d, self.zero = lis[:Constant.HALF_UNIT]

    @staticmethod
    def paras_check(boxes, query_boxes, overlaps, kernel_name):
        """
        Check parameters
        """
        util.check_kernel_name(kernel_name)
        shape_boxes = boxes.get("shape")
        dtype_boxes = boxes.get("dtype").lower()
        util.check_shape_rule(shape_boxes)
        util.check_dtype_rule(dtype_boxes, "float32")
        shape_query_boxes = query_boxes.get("shape")
        dtype_query_boxes = query_boxes.get("dtype").lower()
        util.check_shape_rule(shape_query_boxes)
        util.check_dtype_rule(dtype_query_boxes, "float32")

        shape_overlaps = overlaps.get("shape")
        dtype_overlaps = overlaps.get("dtype").lower()
        util.check_shape_rule(shape_overlaps)
        util.check_dtype_rule(dtype_overlaps, "float32")
        if shape_query_boxes[2] != shape_overlaps[2]:
            raise RuntimeError("Shape unmatch in query_boxes nums")
        if shape_boxes[1] != Constant.INFOS:
            raise RuntimeError("Shape of boxes should be [-1, 7,-1].")
        if shape_query_boxes[1] != Constant.INFOS:
            raise RuntimeError("Shape of query_boxes should be [-1, 7, -1].")
        if shape_query_boxes[2] > Constant.K_LIMIT:
            raise RuntimeError("K's value is over 2000.")
        return [shape_boxes[0], shape_overlaps[1], shape_overlaps[2], dtype_boxes]

    def get_area_of_triangle(self, idx_tmp, idx_current_tmp):
        """
        Calculating triangle area based on vertex coordinates.
        """
        self.b1_x2.set_as(self.corners_ub[idx_tmp])
        self.b1_y2.set_as(self.corners_ub[idx_tmp + Constant.BLOCK])
        self.b1_x3.set_as(self.corners_ub[idx_current_tmp])
        self.b1_y3.set_as(self.corners_ub[idx_current_tmp + Constant.BLOCK])

        self.value.set_as(
            self.b1_x1 * (self.b1_y2 - self.b1_y3) + self.b1_x2 * (self.b1_y3 - self.b1_y1) + self.b1_x3 * (
                    self.b1_y1 - self.b1_y2))

    def sum_area_of_triangles(self):
        """
        Calculate the sum of the areas of the triangles
        """
        self.tik_instance.vec_reduce_add(self.corners_num, self.add_tensor_ub, self.corners_ub, self.work_tensor_ub, 1,
                                         1)
        self.b1_x1.set_as(self.add_tensor_ub[0])
        self.b1_x1.set_as(self.b1_x1 / self.corners_num)
        self.tik_instance.vec_reduce_add(self.corners_num, self.add_tensor_ub, self.corners_ub[Constant.BLOCK],
                                         self.work_tensor_ub, 1, 1)
        self.b1_y1.set_as(self.add_tensor_ub[0])
        self.b1_y1.set_as(self.b1_y1 / self.corners_num)

        self.tik_instance.data_move(self.x_tensor_ub, self.corners_ub, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.y_tensor_ub, self.corners_ub[Constant.BLOCK], 0, 1, 1, 0, 0)

        self.tik_instance.h_sub(self.x_tensor_ub, self.x_tensor_ub, self.b1_x1)
        self.tik_instance.h_sub(self.y_tensor_ub, self.y_tensor_ub, self.b1_y1)
        self.tik_instance.h_div(self.slope_tensor_ub, self.y_tensor_ub, self.x_tensor_ub)
        self.tik_instance.h_cast(self.val_fp16_ub, self.slope_tensor_ub, "none")

        with self.tik_instance.for_range(self.corners_num, Constant.BATCH) as idx:
            self.val_fp16_ub[idx].set_as(self.min_val)

        self.tik_instance.vconcat(self.proposal_ub, self.val_fp16_ub, 1, Constant.VAL_IDX)
        self.tik_instance.vconcat(self.proposal_ub, self.ori_idx_fp16_ub, 1, 0)

        # Sort slopes in descending order
        self.tik_instance.vrpsort16(self.proposal_ub[Constant.BATCH * Constant.BLOCK], self.proposal_ub, 1)
        self.tik_instance.vextract(self.idx_fp16_ub, self.proposal_ub[Constant.BATCH * Constant.BLOCK], 1, 0)

        self.tik_instance.h_cast(self.idx_int32_ub, self.idx_fp16_ub, "round")

        self.idx_left.set_as(0)
        self.idx_right.set_as(0)
        self.cal_area()

    def cal_area(self):
        """
        Calculate the area of a triangle
        """
        idx_current_tmp = self.tik_instance.Scalar("int32")
        with self.tik_instance.for_range(0, self.corners_num) as idx:
            idx_current_tmp.set_as(self.idx_int32_ub[idx])
            self.b1_x.set_as(self.x_tensor_ub[idx_current_tmp])
            with self.tik_instance.if_scope(self.b1_x < 0):
                self.clockwise_idx_int32_ub[self.idx_left].set_as(idx_current_tmp)
                self.idx_left.set_as(self.idx_left + 1)
            with self.tik_instance.elif_scope(self.b1_x > 0):
                self.clockwise_idx_int32_ub[self.idx_right + Constant.BLOCK].set_as(idx_current_tmp)
                self.idx_right.set_as(self.idx_right + 1)
            with self.tik_instance.else_scope():
                self.b1_y.set_as(self.y_tensor_ub[idx_current_tmp])
                with self.tik_instance.if_scope(self.b1_y < 0):
                    self.clockwise_idx_int32_ub[self.idx_left].set_as(idx_current_tmp)
                    self.idx_left.set_as(self.idx_left + 1)
                with self.tik_instance.else_scope():
                    self.clockwise_idx_int32_ub[self.idx_right + Constant.BLOCK].set_as(idx_current_tmp)
                    self.idx_right.set_as(self.idx_right + 1)

        idx_tmp = self.tik_instance.Scalar("int32")
        idx_tmp.set_as(self.clockwise_idx_int32_ub[0])
        with self.tik_instance.for_range(1, self.idx_left) as l_idx:
            idx_current_tmp.set_as(self.clockwise_idx_int32_ub[l_idx])
            self.get_area_of_triangle(idx_tmp, idx_current_tmp)
            self.add_tensor_ub[l_idx].set_as(self.value)
            idx_tmp.set_as(idx_current_tmp)
        with self.tik_instance.for_range(0, self.idx_right) as r_idx:
            idx_current_tmp.set_as(self.clockwise_idx_int32_ub[r_idx + Constant.BLOCK])
            self.get_area_of_triangle(idx_tmp, idx_current_tmp)
            self.add_tensor_ub[r_idx + self.idx_left].set_as(self.value)
            idx_tmp.set_as(idx_current_tmp)
        idx_current_tmp.set_as(self.clockwise_idx_int32_ub[0])
        self.get_area_of_triangle(idx_tmp, idx_current_tmp)
        self.add_tensor_ub[0].set_as(self.value)
        self.tik_instance.h_abs(self.abs_tensor_ub, self.add_tensor_ub)
        self.tik_instance.vec_reduce_add(self.corners_num, self.add_tensor_ub, self.abs_tensor_ub, self.work_tensor_ub,
                                         1, 1)
        self.value.set_as(self.add_tensor_ub[0])

    def record_intersection_point_core(self):
        """
        Each kernel comes up to record the intersection of two cubes
        """
        self.AC_x.set_as(self.b2_x1 - self.b1_x1)
        self.AC_y.set_as(self.b2_y1 - self.b1_y1)
        self.AD_x.set_as(self.b2_x2 - self.b1_x1)
        self.AD_y.set_as(self.b2_y2 - self.b1_y1)
        self.BC_x.set_as(self.b2_x1 - self.b1_x2)
        self.BC_y.set_as(self.b2_y1 - self.b1_y2)
        self.BD_x.set_as(self.b2_x2 - self.b1_x2)
        self.BD_y.set_as(self.b2_y2 - self.b1_y2)

        # Check for intersection between two edges
        self.direct_AC_AD.set_as(self.AC_x * self.AD_y - self.AC_y * self.AD_x)
        self.direct_BC_BD.set_as(self.BC_x * self.BD_y - self.BC_y * self.BD_x)
        with self.tik_instance.if_scope(self.direct_AC_AD * self.direct_BC_BD < 0):
            self.direct_CA_CB.set_as(self.AC_x * self.BC_y - self.AC_y * self.BC_x)
            self.direct_DA_DB.set_as(self.AD_x * self.BD_y - self.AD_y * self.BD_x)
            with self.tik_instance.if_scope(self.direct_CA_CB * self.direct_DA_DB < 0):
                self.tmp_1.set_as(self.b1_x1 * self.b1_y2 - self.b1_y1 * self.b1_x2)
                self.tmp_2.set_as(self.b2_x1 * self.b2_y2 - self.b2_y1 * self.b2_x2)
                self.b1_x1_x2.set_as(self.b1_x1 - self.b1_x2)
                self.b1_y1_y2.set_as(self.b1_y1 - self.b1_y2)
                self.b2_x1_x2.set_as(self.b2_x1 - self.b2_x2)
                self.b2_y1_y2.set_as(self.b2_y1 - self.b2_y2)
                self.denominator.set_as(self.b2_x1_x2 * self.b1_y1_y2 - self.b1_x1_x2 * self.b2_y1_y2)
                self.numerator_x.set_as(self.b1_x1_x2 * self.tmp_2 - self.tmp_1 * self.b2_x1_x2)
                self.numerator_y.set_as(self.b1_y1_y2 * self.tmp_2 - self.tmp_1 * self.b2_y1_y2)
                self.corners_ub[self.corners_num].set_as(self.numerator_x / self.denominator)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.numerator_y / self.denominator)
                self.corners_num.set_as(self.corners_num + 1)

    def record_intersection_point_compute(self):
        """
        The specific process of calculating the intersection point
        """
        self.b1_x2.set_as(self.x2_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y2_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x1.set_as(self.x3_of_boxes_ub[self.b1_offset])
        self.b1_y1.set_as(self.y3_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x2.set_as(self.x4_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y4_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x1.set_as(self.x1_of_boxes_ub[self.b1_offset])
        self.b1_y1.set_as(self.y1_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()

    def record_intersection_point(self, b2_idx):
        """
        Calling function for intersection calculation
        """
        # Calculate the intersection
        self.record_intersection_point_core()
        self.b1_x1.set_as(self.x3_of_boxes_ub[self.b1_offset])
        self.b1_y1.set_as(self.y3_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x2.set_as(self.x4_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y4_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x1.set_as(self.x1_of_boxes_ub[self.b1_offset])
        self.b1_y1.set_as(self.y1_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()

        self.b2_x1.set_as(self.x3_of_boxes_ub[b2_idx])
        self.b2_y1.set_as(self.y3_of_boxes_ub[b2_idx])
        self.record_intersection_point_compute()
        self.b2_x2.set_as(self.x4_of_boxes_ub[b2_idx])
        self.b2_y2.set_as(self.y4_of_boxes_ub[b2_idx])
        self.record_intersection_point_compute()
        self.b2_x1.set_as(self.x1_of_boxes_ub[b2_idx])
        self.b2_y1.set_as(self.y1_of_boxes_ub[b2_idx])
        self.record_intersection_point_compute()

    def record_vertex_point(self, b2_idx):
        """
        Compute the intersection of convex sets
        """
        self.corners_num.set_as(0)
        # func: b1 for input boxes & b2 for input query_boxes
        self.b1_x1.set_as(self.x1_of_boxes_ub[self.b1_offset])
        self.b1_x2.set_as(self.x2_of_boxes_ub[self.b1_offset])
        self.b1_x3.set_as(self.x3_of_boxes_ub[self.b1_offset])
        self.b1_x4.set_as(self.x4_of_boxes_ub[self.b1_offset])
        self.b2_x1.set_as(self.x1_of_boxes_ub[b2_idx])
        self.b2_x2.set_as(self.x2_of_boxes_ub[b2_idx])
        self.b2_x3.set_as(self.x3_of_boxes_ub[b2_idx])
        self.b2_x4.set_as(self.x4_of_boxes_ub[b2_idx])

        self.b1_y1.set_as(self.y1_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y2_of_boxes_ub[self.b1_offset])
        self.b1_y3.set_as(self.y3_of_boxes_ub[self.b1_offset])
        self.b1_y4.set_as(self.y4_of_boxes_ub[self.b1_offset])
        self.b2_y1.set_as(self.y1_of_boxes_ub[b2_idx])
        self.b2_y2.set_as(self.y2_of_boxes_ub[b2_idx])
        self.b2_y3.set_as(self.y3_of_boxes_ub[b2_idx])
        self.b2_y4.set_as(self.y4_of_boxes_ub[b2_idx])
        self.check_first_rectangle()
        self.check_second_rectangle()

    def check_first_rectangle(self):
        """
        Check the vertices of the first rectangle
        """
        # Check if the vertices of the first rectangular box are inside the convex set
        self.AB_x.set_as(self.b2_x2 - self.b2_x1)
        self.AB_y.set_as(self.b2_y2 - self.b2_y1)
        self.AD_x.set_as(self.b2_x4 - self.b2_x1)
        self.AD_y.set_as(self.b2_y4 - self.b2_y1)

        self.AB_AB.set_as(self.AB_x * self.AB_x + self.AB_y * self.AB_y)
        self.AD_AD.set_as(self.AD_x * self.AD_x + self.AD_y * self.AD_y)

        self.AP_x.set_as(self.b1_x1 - self.b2_x1)
        self.AP_y.set_as(self.b1_y1 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB >= 0, self.AP_AB <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD >= 0, self.AP_AD <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b1_x1)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y1)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x2 - self.b2_x1)
        self.AP_y.set_as(self.b1_y2 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB >= 0, self.AP_AB <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD >= 0, self.AP_AD <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b1_x2)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y2)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x3 - self.b2_x1)
        self.AP_y.set_as(self.b1_y3 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB >= 0, self.AP_AB <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD >= 0, self.AP_AD <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b1_x3)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y3)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x4 - self.b2_x1)
        self.AP_y.set_as(self.b1_y4 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB >= 0, self.AP_AB <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD >= 0, self.AP_AD <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b1_x4)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y4)
                self.corners_num.set_as(self.corners_num + 1)

    def check_second_rectangle(self):
        """
        Check the vertices of the second rectangle
        """
        self.AB_x.set_as(self.b1_x2 - self.b1_x1)
        self.AB_y.set_as(self.b1_y2 - self.b1_y1)
        self.AD_x.set_as(self.b1_x4 - self.b1_x1)
        self.AD_y.set_as(self.b1_y4 - self.b1_y1)
        self.AB_AB.set_as(self.AB_x * self.AB_x + self.AB_y * self.AB_y)
        self.AD_AD.set_as(self.AD_x * self.AD_x + self.AD_y * self.AD_y)

        # Check if the vertices of the second rectangular box are inside the convex set
        self.AP_x.set_as(self.b2_x1 - self.b1_x1)
        self.AP_y.set_as(self.b2_y1 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB >= 0, self.AP_AB <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD >= 0, self.AP_AD <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b2_x1)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y1)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x2 - self.b1_x1)
        self.AP_y.set_as(self.b2_y2 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB >= 0, self.AP_AB <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD >= 0, self.AP_AD <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b2_x2)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y2)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x3 - self.b1_x1)
        self.AP_y.set_as(self.b2_y3 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB >= 0, self.AP_AB <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD >= 0, self.AP_AD <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b2_x3)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y3)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x4 - self.b1_x1)
        self.AP_y.set_as(self.b2_y4 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB >= 0, self.AP_AB <= self.AB_AB)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD >= 0, self.AP_AD <= self.AD_AD)):
                self.corners_ub[self.corners_num].set_as(self.b2_x4)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y4)
                self.corners_num.set_as(self.corners_num + 1)

    def get_effective_depth(self, task_idx, current_batch):
        """
        Get the effective depth of the intersecting volume
        """
        self.tik_instance.data_move(
            self.d_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.D_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            self.z_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.Z_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        if self.b1_batch == 1:
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.D_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.d_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.Z_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.z_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        else:
            self.tik_instance.data_move(
                self.d_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.D_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                self.z_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.Z_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
        self.tik_instance.h_mul(self.half_d_of_boxes_ub, self.d_of_boxes_ub, self.half)
        self.tik_instance.h_sub(self.z_sub_d_boxes_ub, self.z_of_boxes_ub, self.half_d_of_boxes_ub)
        self.tik_instance.h_add(self.z_add_d_boxes_ub, self.z_of_boxes_ub, self.half_d_of_boxes_ub)

    def trans_boxes(self, task_idx, current_batch):
        """
        Calculate the coordinates of the rotated box
        """
        # theta
        self.tik_instance.data_move(
            self.t_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.T_IDX + current_batch * Constant.INFOS)], 0,
            1, self.repeats, 0, 0)
        if self.b1_batch == 1:
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.T_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.t_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        else:
            self.tik_instance.data_move(
                self.t_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.T_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)

        self.tik_instance.h_mul(self.radian_t_of_boxes_ub, self.t_of_boxes_ub, self.radian)
        self.tik_instance.h_sin(self.sin_t_of_boxes_ub, self.radian_t_of_boxes_ub)
        self.tik_instance.h_cos(self.cos_t_of_boxes_ub, self.radian_t_of_boxes_ub)
        self.data_move_with_w_and_h(current_batch, task_idx)
        self.data_move_with_x_and_y(current_batch, task_idx)
        self.cal_coordinate_with_rotate()

    def data_move_with_w_and_h(self, current_batch, task_idx):
        """
        Move the width and height of the cuboid
        """
        self.tik_instance.data_move(
            self.w_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.W_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            self.h_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.H_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)

        if self.b1_batch == 1:
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.W_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.w_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.H_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.h_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        else:
            self.tik_instance.data_move(
                self.w_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.W_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                self.h_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.H_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
        self.tik_instance.h_mul(self.half_w_of_boxes_ub, self.w_of_boxes_ub, self.half)
        self.tik_instance.h_mul(self.half_h_of_boxes_ub, self.h_of_boxes_ub, self.half)

    def data_move_with_x_and_y(self, current_batch, task_idx):
        """
        Move the x and y coordinates of the center point of the cuboid
        """
        self.tik_instance.data_move(
            self.x_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.X_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            self.y_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.Y_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)

        if self.b1_batch == 1:
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.X_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.x_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.Y_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.y_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        else:
            self.tik_instance.data_move(
                self.x_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.X_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                self.y_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.Y_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)

    def cal_coordinate_with_rotate(self):
        """
        Specifically calculate the coordinates after the cuboid is rotated
        """
        self.tik_instance.h_mul(self.half_w_cos_of_boxes_ub, self.cos_t_of_boxes_ub, self.half_w_of_boxes_ub)
        self.tik_instance.h_mul(self.half_w_sin_of_boxes_ub, self.sin_t_of_boxes_ub, self.half_w_of_boxes_ub)
        self.tik_instance.h_mul(self.half_h_cos_of_boxes_ub, self.cos_t_of_boxes_ub, self.half_h_of_boxes_ub)
        self.tik_instance.h_mul(self.half_h_sin_of_boxes_ub, self.sin_t_of_boxes_ub, self.half_h_of_boxes_ub)

        self.tik_instance.h_sub(self.x_sub_w_of_boxes_ub, self.x_of_boxes_ub, self.half_w_cos_of_boxes_ub)
        self.tik_instance.h_sub(self.y_sub_w_of_boxes_ub, self.y_of_boxes_ub, self.half_w_sin_of_boxes_ub)
        self.tik_instance.h_add(self.x_add_w_of_boxes_ub, self.x_of_boxes_ub, self.half_w_cos_of_boxes_ub)
        self.tik_instance.h_add(self.y_add_w_of_boxes_ub, self.y_of_boxes_ub, self.half_w_sin_of_boxes_ub)

        self.tik_instance.h_sub(self.x1_of_boxes_ub, self.x_sub_w_of_boxes_ub, self.half_h_sin_of_boxes_ub)
        self.tik_instance.h_add(self.y1_of_boxes_ub, self.y_sub_w_of_boxes_ub, self.half_h_cos_of_boxes_ub)

        self.tik_instance.h_sub(self.x2_of_boxes_ub, self.x_add_w_of_boxes_ub, self.half_h_sin_of_boxes_ub)
        self.tik_instance.h_add(self.y2_of_boxes_ub, self.y_add_w_of_boxes_ub, self.half_h_cos_of_boxes_ub)

        self.tik_instance.h_add(self.x3_of_boxes_ub, self.x_add_w_of_boxes_ub, self.half_h_sin_of_boxes_ub)
        self.tik_instance.h_sub(self.y3_of_boxes_ub, self.y_add_w_of_boxes_ub, self.half_h_cos_of_boxes_ub)

        self.tik_instance.h_add(self.x4_of_boxes_ub, self.x_sub_w_of_boxes_ub, self.half_h_sin_of_boxes_ub)
        self.tik_instance.h_sub(self.y4_of_boxes_ub, self.y_sub_w_of_boxes_ub, self.half_h_cos_of_boxes_ub)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def data_init(self):
        """
        Data initialization overall function
        """
        # Tensor
        self.idx_fp16_ub = self.tik_instance.Tensor("float16", [Constant.BATCH], name="idx_fp16_ub",
                                                    scope=tik.scope_ubuf)
        self.ori_idx_fp16_ub = self.tik_instance.Tensor("float16", [Constant.BATCH], name="ori_idx_fp16_ub",
                                                        scope=tik.scope_ubuf)

        self.box_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="box_ub", scope=tik.scope_ubuf)
        self.overlap_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="overlap_ub",
                                                   scope=tik.scope_ubuf)

        self.x_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.y_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.z_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="z_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="w_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.h_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="h_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.d_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="h_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.half_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="half_w_of_boxes_ub",
                                                           scope=tik.scope_ubuf)
        self.half_h_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="half_h_of_boxes_ub",
                                                           scope=tik.scope_ubuf)
        self.half_d_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="half_d_of_boxes_ub",
                                                           scope=tik.scope_ubuf)
        self.z_sub_d_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="z_sub_h_boxes_ub",
                                                           scope=tik.scope_ubuf)
        self.z_add_d_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="z_add_d_boxes_ub",
                                                         scope=tik.scope_ubuf)
        self.volume_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="volume_of_boxes_ub",
                                                         scope=tik.scope_ubuf)

        self.t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="t_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.radian_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="radian_t_of_boxes_ub",
                                                             scope=tik.scope_ubuf)
        self.data_init_with_rectangle_tensor()
        self.data_init_with_mid_tensor()
        self.data_init_with_point_scalar()
        self.data_init_with_line_scalar()

    def data_init_with_rectangle_tensor(self):
        """
        Rectangle variable data initialization
        """
        self.cos_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="cos_t_of_boxes_ub",
                                                          scope=tik.scope_ubuf)
        self.sin_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="sin_t_of_boxes_ub",
                                                          scope=tik.scope_ubuf)
        self.half_w_cos_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                               name="half_w_cos_of_boxes_ub", scope=tik.scope_ubuf)
        self.half_w_sin_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                               name="half_w_sin_of_boxes_ub", scope=tik.scope_ubuf)
        self.half_h_cos_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                               name="half_h_cos_of_boxes_ub", scope=tik.scope_ubuf)
        self.half_h_sin_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                               name="half_h_sin_of_boxes_ub", scope=tik.scope_ubuf)

        self.x_sub_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_sub_w_of_boxes_ub",
                                                            scope=tik.scope_ubuf)
        self.y_sub_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_sub_w_of_boxes_ub",
                                                            scope=tik.scope_ubuf)
        self.x_add_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_add_w_of_boxes_ub",
                                                            scope=tik.scope_ubuf)
        self.y_add_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_add_w_of_boxes_ub",
                                                            scope=tik.scope_ubuf)

        self.x1_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x1_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.x2_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x2_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.x3_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x3_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.x4_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x4_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.y1_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y1_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.y2_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y2_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.y3_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y3_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.y4_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y4_of_boxes_ub",
                                                       scope=tik.scope_ubuf)

    def data_init_with_mid_tensor(self):
        """
        Intermediate variable data initialization
        """
        self.add_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="add_tensor_ub",
                                                      scope=tik.scope_ubuf)
        self.abs_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="abs_tensor_ub",
                                                      scope=tik.scope_ubuf)
        self.tmp_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="tmp_tensor_ub",
                                                      scope=tik.scope_ubuf)
        self.work_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="work_tensor_ub",
                                                       scope=tik.scope_ubuf)

        self.corners_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="corners_ub",
                                                   scope=tik.scope_ubuf)
        self.val_fp16_ub = self.tik_instance.Tensor("float16", [Constant.BATCH], name="val_fp16_ub",
                                                    scope=tik.scope_ubuf)
        self.idx_int32_ub = self.tik_instance.Tensor("int32", [Constant.BATCH], name="idx_int32_ub",
                                                     scope=tik.scope_ubuf)
        self.proposal_ub = self.tik_instance.Tensor("float16", [2, Constant.BATCH, Constant.BLOCK],
                                                    name="proposal_ub", scope=tik.scope_ubuf)

        self.x_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="x_tensor_ub",
                                                    scope=tik.scope_ubuf)
        self.y_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="y_tensor_ub",
                                                    scope=tik.scope_ubuf)

        self.slope_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="slope_tensor_ub",
                                                        scope=tik.scope_ubuf)
        self.clockwise_idx_int32_ub = self.tik_instance.Tensor("int32", [Constant.BATCH],
                                                               name="clockwise_idx_int32_ub", scope=tik.scope_ubuf)

    def data_init_with_point_scalar(self):
        """
        point vector initialization
        """
        self.idx_fp32 = self.tik_instance.Scalar(self.dtype, init_value=0)
        self.min_val = self.tik_instance.Scalar('float16', init_value=Constant.MIN_VAL)
        self.half = self.tik_instance.Scalar(self.dtype, init_value=0.5)
        self.radian = self.tik_instance.Scalar(self.dtype, init_value=1)
        self.value = self.tik_instance.Scalar(self.dtype)
        self.w_value = self.tik_instance.Scalar(self.dtype)
        self.h_value = self.tik_instance.Scalar(self.dtype)
        self.d_value = self.tik_instance.Scalar(self.dtype)
        self.valid_box_num = self.tik_instance.Scalar('int32')
        self.mov_repeats = self.tik_instance.Scalar('int32')
        self.corners_num = self.tik_instance.Scalar("int32")
        self.idx_right = self.tik_instance.Scalar("int32")
        self.idx_left = self.tik_instance.Scalar("int32")
        self.b1_offset = self.tik_instance.Scalar("int32")

        self.b1_x = self.tik_instance.Scalar(self.dtype)
        self.b1_y = self.tik_instance.Scalar(self.dtype)
        self.b2_x = self.tik_instance.Scalar(self.dtype)
        self.b2_y = self.tik_instance.Scalar(self.dtype)
        self.b1_x1 = self.tik_instance.Scalar(self.dtype)
        self.b1_y1 = self.tik_instance.Scalar(self.dtype)
        self.b2_x1 = self.tik_instance.Scalar(self.dtype)
        self.b2_y1 = self.tik_instance.Scalar(self.dtype)
        self.b1_x2 = self.tik_instance.Scalar(self.dtype)
        self.b1_y2 = self.tik_instance.Scalar(self.dtype)
        self.b2_x2 = self.tik_instance.Scalar(self.dtype)
        self.b2_y2 = self.tik_instance.Scalar(self.dtype)
        self.b1_x3 = self.tik_instance.Scalar(self.dtype)
        self.b1_y3 = self.tik_instance.Scalar(self.dtype)
        self.b2_x3 = self.tik_instance.Scalar(self.dtype)
        self.b2_y3 = self.tik_instance.Scalar(self.dtype)

        self.b1_x4 = self.tik_instance.Scalar(self.dtype)
        self.b1_y4 = self.tik_instance.Scalar(self.dtype)
        self.b2_x4 = self.tik_instance.Scalar(self.dtype)
        self.b2_y4 = self.tik_instance.Scalar(self.dtype)
        self.b1_x1_x2 = self.tik_instance.Scalar(self.dtype)
        self.b1_y1_y2 = self.tik_instance.Scalar(self.dtype)
        self.b2_x1_x2 = self.tik_instance.Scalar(self.dtype)
        self.b2_y1_y2 = self.tik_instance.Scalar(self.dtype)

    def data_init_with_line_scalar(self):
        """
        Line vector initialization
        """
        self.AB_x = self.tik_instance.Scalar(self.dtype)
        self.AB_y = self.tik_instance.Scalar(self.dtype)
        self.AC_x = self.tik_instance.Scalar(self.dtype)
        self.AC_y = self.tik_instance.Scalar(self.dtype)
        self.AD_x = self.tik_instance.Scalar(self.dtype)
        self.AD_y = self.tik_instance.Scalar(self.dtype)
        self.AP_x = self.tik_instance.Scalar(self.dtype)
        self.AP_y = self.tik_instance.Scalar(self.dtype)

        self.AB_AB = self.tik_instance.Scalar(self.dtype)
        self.AD_AD = self.tik_instance.Scalar(self.dtype)
        self.AP_AB = self.tik_instance.Scalar(self.dtype)
        self.AP_AD = self.tik_instance.Scalar(self.dtype)

        self.BC_x = self.tik_instance.Scalar(self.dtype)
        self.BC_y = self.tik_instance.Scalar(self.dtype)
        self.BD_x = self.tik_instance.Scalar(self.dtype)
        self.BD_y = self.tik_instance.Scalar(self.dtype)

        self.direct_AC_AD = self.tik_instance.Scalar(self.dtype)
        self.direct_BC_BD = self.tik_instance.Scalar(self.dtype)
        self.direct_CA_CB = self.tik_instance.Scalar(self.dtype)
        self.direct_DA_DB = self.tik_instance.Scalar(self.dtype)

        self.tmp_1 = self.tik_instance.Scalar(self.dtype)
        self.tmp_2 = self.tik_instance.Scalar(self.dtype)
        self.denominator = self.tik_instance.Scalar(self.dtype)
        self.numerator_x = self.tik_instance.Scalar(self.dtype)
        self.numerator_y = self.tik_instance.Scalar(self.dtype)
        self.max_of_min = self.tik_instance.Scalar(self.dtype)
        self.min_of_max = self.tik_instance.Scalar(self.dtype)
        self.real_d = self.tik_instance.Scalar(self.dtype)
        self.zero = self.tik_instance.Scalar(self.dtype, init_value=0)

    def compute_core(self, task_idx):
        """
        core computing
        """
        self.data_init()
        inter_volume = self.tik_instance.Scalar(self.dtype, init_value=0)
        b1_volume = self.tik_instance.Scalar(self.dtype)
        b2_volume = self.tik_instance.Scalar(self.dtype)
        b1_min = self.tik_instance.Scalar(self.dtype)
        b2_min = self.tik_instance.Scalar(self.dtype)
        b1_max = self.tik_instance.Scalar(self.dtype)
        b2_max = self.tik_instance.Scalar(self.dtype)
        with self.tik_instance.for_range(0, Constant.BLOCK) as i:
            self.ori_idx_fp16_ub[i].set_as(self.idx_fp32)
            self.idx_fp32.set_as(self.idx_fp32 + 1)
        with self.tik_instance.for_range(0, self.batch) as current_batch:
            self.trans_boxes(task_idx, current_batch)
            self.get_effective_depth(task_idx, current_batch)
            self.valid_box_num.set_as(0)
            self.tik_instance.h_mul(self.volume_of_boxes_ub, self.h_of_boxes_ub, self.w_of_boxes_ub)
            self.tik_instance.h_mul(self.volume_of_boxes_ub, self.volume_of_boxes_ub, self.d_of_boxes_ub)
            # record the valid query_boxes's num
            with self.tik_instance.for_range(0, self.k) as idx:
                self.w_value.set_as(self.w_of_boxes_ub[idx])
                self.h_value.set_as(self.h_of_boxes_ub[idx])
                self.d_value.set_as(self.d_of_boxes_ub[idx])
                with self.tik_instance.if_scope(self.w_value * self.h_value * self.d_value > 0):
                    self.valid_box_num.set_as(self.valid_box_num + 1)
            self.mov_repeats.set_as((self.valid_box_num + Constant.BLOCK - 1) // Constant.BLOCK)
            lis = [b1_volume, b2_volume, b1_min, b1_max, b2_min, b2_max, inter_volume]
            self.main_compute_per_core(lis, task_idx, current_batch)

    def main_compute_per_core(self, lis, task_idx, current_batch):
        """
        Core Specific Computing
        """
        b1_volume, b2_volume, b1_min, b1_max, b2_min, b2_max, inter_volume = lis
        with self.tik_instance.for_range(0, self.b1_batch) as b1_idx:
            self.tik_instance.vec_dup(Constant.BLOCK, self.overlap_ub, 0, self.mov_repeats, 1)
            self.b1_offset.set_as(self.k_align - self.b1_batch + b1_idx)
            b1_volume.set_as(self.volume_of_boxes_ub[self.b1_offset])
            b1_min.set_as(self.z_sub_d_boxes_ub[self.b1_offset])
            b1_max.set_as(self.z_add_d_boxes_ub[self.b1_offset])
            with self.tik_instance.for_range(0, self.valid_box_num) as b2_idx:
                self.record_vertex_point(b2_idx)
                self.record_intersection_point(b2_idx)
                b2_volume.set_as(self.volume_of_boxes_ub[b2_idx])
                b2_min.set_as(self.z_sub_d_boxes_ub[b2_idx])
                b2_max.set_as(self.z_add_d_boxes_ub[b2_idx])
                with self.tik_instance.if_scope(b2_min > b1_min):
                    self.max_of_min.set_as(b2_min)
                with self.tik_instance.else_scope():
                    self.max_of_min.set_as(b1_min)
                with self.tik_instance.if_scope(b2_max > b1_max):
                    self.min_of_max.set_as(b1_max)
                with self.tik_instance.else_scope():
                    self.min_of_max.set_as(b2_max)
                with self.tik_instance.if_scope(self.min_of_max - self.max_of_min > self.zero):
                    self.real_d.set_as(self.min_of_max - self.max_of_min)
                with self.tik_instance.else_scope():
                    self.real_d.set_as(self.zero)
                with self.tik_instance.if_scope(self.corners_num == 3):
                    self.b1_x1.set_as(self.corners_ub[0])
                    self.b1_y1.set_as(self.corners_ub[Constant.BLOCK])
                    self.get_area_of_triangle(1, 2)
                    with self.tik_instance.if_scope(self.value > 0):
                        inter_volume.set_as(self.value / 2)
                    with self.tik_instance.else_scope():
                        inter_volume.set_as(-1 * self.value / 2)
                with self.tik_instance.if_scope(self.corners_num > 3):
                    self.sum_area_of_triangles()
                    inter_volume.set_as(self.value / 2)
                with self.tik_instance.if_scope(self.corners_num == 0):
                    inter_volume.set_as(0)
                inter_volume.set_as(self.real_d * inter_volume)
                with self.tik_instance.if_scope(b1_volume + b2_volume - inter_volume > 0):
                    self.overlap_ub[b2_idx].set_as(
                        inter_volume / (b1_volume + b2_volume - inter_volume + Constant.EPSILON))
            self.tik_instance.data_move(
                self.iou_gm[self.k * (task_idx * self.b1_batch + b1_idx + current_batch * self.n)],
                self.overlap_ub, 0, 1, self.mov_repeats, 0, 0)

    def compute(self):
        """
        Calculate the total interface
        """
        self.tik_instance.set_atomic_add(1)
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.compute_core(i + j * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.compute_core(self.batch_num_per_aicore * self.used_aicore_num + i)
        self.tik_instance.set_atomic_add(0)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.boxes_gm, self.query_boxes_gm],
                                   outputs=[self.iou_gm])
        return self.tik_instance


# 'pylint:disable=too-many-arguments, disable=too-many-statements
@tbe_platform.fusion_manager.fusion_manager.register("iou_3d")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def iou_3d(boxes, query_boxes, iou, kernel_name="iou_3d"):
    """
    Function: compute the 3d iou.
    Modify : 2022-05-31

    Init base parameters
    Parameters
    ----------
    boxes: dict
        data of input
    query_boxes: dict
        data of input
    iou: dict
        data of output
    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = Iou3D(boxes, query_boxes, iou, kernel_name)
    return op_obj.compute()
