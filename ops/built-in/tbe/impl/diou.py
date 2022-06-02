# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
diou
"""


from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector


# pylint: disable=too-few-public-methods,invalid-name,unused-variable
class Constant:
    """
    The class for constant.
    """
    FP16_ELIMENTS_BLOCK = 16
    FP32_ELIMENTS_BLOCK = 8
    GTBOX_SEGMENT = 2048
    BBOX_SEGMENT = 2048


def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    ceil_int = (int1 + int2 - 1) // int2
    return ceil_int


# pylint: disable=too-many-instance-attributes,too-many-lines
class DIoU():
    """Function: use to finish Iou main functions
    """

    # pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, bboxes, gtboxes, trans, is_cross, mode):
        self.bboxes_shape = bboxes.get("shape")
        self.bboxes_dtype = bboxes.get("dtype").lower()
        self.gtboxes_shape = gtboxes.get("shape")
        self.gtboxes_dtype = gtboxes.get("dtype").lower()
        self.gtboxes_num = self.gtboxes_shape[1]
        self.dtype = self.bboxes_dtype
        self.trans = trans
        self.is_cross = is_cross
        self.mode = mode.lower()
        self.tik_instance = tik.Tik()
        self.core_num = tik.Dprofile().get_aicore_num()
        self.product = tbe_platform.api_check_support("tik.vdiv", "float32")
        # input and output tensor in gm
        self.diou_shape = [1, self.gtboxes_shape[1]]
        self.bboxes_gm = self.tik_instance.Tensor(self.bboxes_dtype, self.bboxes_shape,
                                                  name="bboxes_gm", scope=tik.scope_gm)
        self.gtboxes_gm = self.tik_instance.Tensor(self.gtboxes_dtype, self.gtboxes_shape,
                                                   name="gtboxes_gm", scope=tik.scope_gm)
        self.diou_gm = self.tik_instance.Tensor(self.bboxes_dtype, self.diou_shape, name="diou_gm", scope=tik.scope_gm)

        # init attr in objext
        self.point_per_core = self.core_tail_num = self.bb_ub_segment = 0
        self.bboxes_x0 = self.bboxes_x1 = self.bboxes_y0 = self.bboxes_y1 = None
        self.gtboxes_x0 = self.gtboxes_x1 = self.gtboxes_y0 = self.gtboxes_y1 = None
        self.inter_area_x0 = self.inter_area_x1 = self.inter_area_y0 = self.inter_area_y1 = None
        self.outer_area_x0 = self.outer_area_x1 = self.outer_area_y0 = self.outer_area_y1 = None
        self.in_square = self.out_square = self.div_rec_1 = self.div_rec_2 = None
        self.area_y1_y0 = self.sum_y1_y0 = self.gtboxes_area_ub = self.out_ub = None
        self.bboxes_area_ub = self.inter_area_ub = self.zero_ub = None
        block_parm_dict = {"float16": Constant.FP16_ELIMENTS_BLOCK, "float32": Constant.FP32_ELIMENTS_BLOCK}
        self.min_point_per_core = block_parm_dict.get(self.bboxes_dtype)
        self.eliments_per_block = block_parm_dict.get(self.bboxes_dtype)
        if self.bboxes_dtype == "float32":
            self.bb_ub_segment = Constant.BBOX_SEGMENT // 2
        else:
            self.bb_ub_segment = Constant.BBOX_SEGMENT
        self.max_eliments = block_parm_dict.get(self.bboxes_dtype) * 8

    # pylint: disable=too-many-locals,too-many-branches,too-many-lines,too-many-statements
    def diou_process(self):
        self.point_per_core = _get_ceil_int(self.bboxes_shape[1], self.core_num)
        if self.point_per_core < self.min_point_per_core:
            self.point_per_core = self.min_point_per_core
        self.point_per_core =  _get_ceil_int(self.point_per_core, self.min_point_per_core) * self.min_point_per_core
        self.core_tail_num = self.bboxes_shape[1] % self.point_per_core
        self.core_num = _get_ceil_int(self.bboxes_shape[1], self.point_per_core)
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as _core_id:
            # calcu gt area
            bb_tail = self.point_per_core % self.bb_ub_segment
            bb_loop = self.point_per_core // self.bb_ub_segment
            if (0 < bb_tail < self.min_point_per_core) and bb_loop != 0:
                bb_tail_offset = bb_loop * self.bb_ub_segment + bb_tail - self.min_point_per_core
                bb_tail = self.min_point_per_core
            elif bb_tail % self.min_point_per_core != 0 and bb_loop != 0:
                bb_tail_offset = bb_loop * self.bb_ub_segment + (bb_tail % self.min_point_per_core) - \
                                 self.min_point_per_core
                bb_tail = (bb_tail // self.min_point_per_core + 1) * self.min_point_per_core
            else:
                bb_tail_offset = bb_loop * self.bb_ub_segment
            dst_gm_offset = 0
            repeat_time_max = self.bb_ub_segment // self.max_eliments
            if self.core_tail_num != 0:
                with self.tik_instance.if_scope(_core_id == (self.core_num - 1)):
                    if self.core_num != 1:
                        dst_gm_offset = self.point_per_core * _core_id - self.point_per_core + self.core_tail_num
                with self.tik_instance.else_scope():
                    dst_gm_offset = self.point_per_core * _core_id
            else:
                dst_gm_offset = self.point_per_core * _core_id
            self.diou_process_one_loop(bb_loop, dst_gm_offset, repeat_time_max, bb_tail, bb_tail_offset)
    
    def diou_process_one_loop(self, bb_loop, dst_gm_offset, repeat_time_max, bb_tail, bb_tail_offset):
        with self.tik_instance.for_range(0, bb_loop) as bb_loop_index:
            self._run_segment(self.max_eliments, repeat_time_max, bb_loop_index * self.bb_ub_segment + dst_gm_offset)
        if bb_tail != 0:
            if bb_tail // self.max_eliments > 0:
                self._run_segment(self.max_eliments, bb_tail // self.max_eliments, bb_tail_offset + dst_gm_offset)
            gm_point_offset = bb_tail_offset + dst_gm_offset + bb_tail // self.max_eliments * self.max_eliments
            if bb_tail % self.max_eliments > 0:
                self._run_segment(bb_tail % self.max_eliments, 1, gm_point_offset)

    def run_tik(self, kernel_name):
        self.diou_process()
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.bboxes_gm, self.gtboxes_gm],
                                   outputs=[self.diou_gm])
        return self.tik_instance

    def data_move_in_and_trans(self, mask, repeat_time, one_loop_shape, gm_offset, nbust):
        boxes_xy = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "boxes_xy")
        boxes_wh = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "boxes_wh")

        self.tik_instance.data_move(boxes_xy, self.gtboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 2], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.gtboxes_x0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.gtboxes_x1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(boxes_xy, self.gtboxes_gm[gm_offset + self.bboxes_shape[1]], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 3], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.gtboxes_y0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.gtboxes_y1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(boxes_xy, self.bboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 2], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.bboxes_x0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.bboxes_x1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(boxes_xy, self.bboxes_gm[gm_offset + self.bboxes_shape[1]], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 3], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.bboxes_y0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.bboxes_y1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)

    def data_move_in(self, gm_offset, nbust):
        self.tik_instance.data_move(self.gtboxes_x0, self.gtboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_y0, self.gtboxes_gm[gm_offset + self.bboxes_shape[1]],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_x1, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 2],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_y1, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 3],
                                    0, 1, nbust, 0, 0)

        self.tik_instance.data_move(self.bboxes_x0, self.bboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_y0, self.bboxes_gm[gm_offset + self.bboxes_shape[1]],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_x1, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 2],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_y1, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 3],
                                    0, 1, nbust, 0, 0)

    def get_inter_outer_area(self):
        self.tik_instance.h_max(self.inter_area_x0, self.bboxes_x0, self.gtboxes_x0)
        self.tik_instance.h_max(self.inter_area_y0, self.bboxes_y0, self.gtboxes_y0)
        self.tik_instance.h_min(self.inter_area_x1, self.bboxes_x1, self.gtboxes_x1)
        self.tik_instance.h_min(self.inter_area_y1, self.bboxes_y1, self.gtboxes_y1)
        
        self.tik_instance.h_min(self.outer_area_x0, self.bboxes_x0, self.gtboxes_x0)
        self.tik_instance.h_min(self.outer_area_y0, self.bboxes_y0, self.gtboxes_y0)
        self.tik_instance.h_max(self.outer_area_x1, self.bboxes_x1, self.gtboxes_x1)
        self.tik_instance.h_max(self.outer_area_y1, self.bboxes_y1, self.gtboxes_y1)

    # pylint: disable=too-many-arguments
    def calcu_area(self, mask, repeat_time, area_ub, inter_mode=False, gt_mode=False):
        if inter_mode:
            x0_ub = self.inter_area_x0
            x1_ub = self.inter_area_x1
            y0_ub = self.inter_area_y0
            y1_ub = self.inter_area_y1
        elif gt_mode:
            x0_ub = self.gtboxes_x0
            x1_ub = self.gtboxes_x1
            y0_ub = self.gtboxes_y0
            y1_ub = self.gtboxes_y1
        else:
            x0_ub = self.bboxes_x0
            x1_ub = self.bboxes_x1
            y0_ub = self.bboxes_y0
            y1_ub = self.bboxes_y1
        # cala x1 - x0
        self.tik_instance.vsub(mask, self.area_y1_y0, y1_ub, y0_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, area_ub, x1_ub, x0_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        if inter_mode:
            self.tik_instance.vmax(mask, area_ub, self.zero_ub, area_ub, repeat_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vmax(mask, self.area_y1_y0, self.zero_ub, self.area_y1_y0, repeat_time, 1, 0, 1, 8, 0, 8)
        else:
            self.tik_instance.vadds(mask, area_ub, area_ub, 1e-16, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vadds(mask, self.area_y1_y0, self.area_y1_y0, 1e-16, repeat_time, 1, 1, 8, 8)
        
        self.tik_instance.vmul(mask, area_ub, self.area_y1_y0, area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
    
    def calcu_in_square(self, mask, repeat_time):
        self.tik_instance.vadd(mask, self.sum_y1_y0, self.bboxes_y0, self.bboxes_y1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.sum_y1_y0, self.sum_y1_y0, self.gtboxes_y0, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.sum_y1_y0, self.sum_y1_y0, self.gtboxes_y1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.sum_y1_y0, self.sum_y1_y0, self.sum_y1_y0, repeat_time, 1, 1, 1, 8, 8, 8)
        
        self.tik_instance.vadd(mask, self.in_square, self.bboxes_x0, self.bboxes_x1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.in_square, self.in_square, self.gtboxes_x0, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.in_square, self.in_square, self.gtboxes_x1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.in_square, self.in_square, self.in_square, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(mask, self.in_square, self.in_square, self.sum_y1_y0, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.in_square, self.in_square, 0.25, repeat_time, 1, 1, 8, 8)
    
    def calcu_out_square(self, mask, repeat_time):
        self.tik_instance.vsub(mask, self.out_square, self.outer_area_x1, self.outer_area_x0,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.area_y1_y0, self.outer_area_y1, self.outer_area_y0,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.out_square, self.out_square, self.out_square, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.area_y1_y0, self.area_y1_y0, self.area_y1_y0, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.out_square, self.out_square, self.area_y1_y0, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(mask, self.out_square, self.out_square, 1e-16, repeat_time, 1, 1, 8, 8)
   
    def _apply_all_ub(self, one_loop_shape):
        self.bboxes_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_x0")
        self.gtboxes_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_x0")
        self.bboxes_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_x1")
        self.gtboxes_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_x1")
        self.inter_area_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_x0")
        self.outer_area_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_x0")
        self.inter_area_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_x1")
        self.outer_area_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_x1")
        self.bboxes_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_y0")
        self.gtboxes_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_y0")
        self.bboxes_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_y1")
        self.gtboxes_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_y1")
        self.inter_area_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_y0")
        self.outer_area_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_y0")
        self.inter_area_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_y1")
        self.outer_area_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_y1")
        self.area_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "area_y1_y0")
        self.sum_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "sum_y1_y0")
        self.in_square = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "in_square")
        self.out_square = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "out_square")
        self.gtboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_area_ub")
        self.bboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_area_ub")
        self.inter_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_ub")
        self.zero_ub = _apply_mem(self.tik_instance, self.dtype, [self.eliments_per_block], "zero_ub")
        self.tik_instance.vector_dup(self.eliments_per_block, self.zero_ub, 0.0, 1, 1, 8)
        self.div_rec_1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "div_rec_1")
        self.div_rec_2 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "div_rec_2")
        self.out_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "out_ub")

    # pylint: disable=too-many-locals,too-many-branches,too-many-lines
    def _run_segment(self, mask, repeat_time, gm_offset):
        """
        do a segment of bbox compute
        """
        one_loop_shape = mask * repeat_time
        self._apply_all_ub(one_loop_shape)
        nbust = _get_ceil_int(one_loop_shape, self.eliments_per_block)

        # copy gm to ub
        if not self.trans:
            self.data_move_in(gm_offset, nbust)
        else:
            self.data_move_in_and_trans(mask, repeat_time, one_loop_shape, gm_offset, nbust)

        # calcu bboxes area
        self.calcu_area(mask, repeat_time, self.bboxes_area_ub)

        # calcu gtboxes area
        self.calcu_area(mask, repeat_time, self.gtboxes_area_ub, gt_mode=True)

        # vmin vmax: get inter x0 x1 y0 y1, outer x0 x1 y0 y1
        self.get_inter_outer_area()
        
        # calcu inter area
        self.calcu_area(mask, repeat_time, self.inter_area_ub, inter_mode=True)

        self.calcu_out_square(mask, repeat_time)
        self.calcu_in_square(mask, repeat_time)

        if self.mode == "iou":
            self.tik_instance.vadd(mask, self.out_ub, self.bboxes_area_ub,
                                   self.gtboxes_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(mask, self.out_ub, self.out_ub, self.inter_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        elif self.mode == "iof":
            self.tik_instance.data_move(self.out_ub, self.gtboxes_area_ub, 0, 1, (nbust - 1) // 4 + 1, 0, 0)

        if self.product is True:
            self.tik_instance.vdiv(mask, self.out_ub, self.inter_area_ub, self.out_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vdiv(mask, self.out_square, self.in_square,
                                   self.out_square, repeat_time, 1, 1, 1, 8, 8, 8)
        else:
            # for mini
            self._rev_div(mask, repeat_time, self.out_ub, self.inter_area_ub, self.out_ub)
            self._rev_div(mask, repeat_time, self.out_square, self.in_square, self.out_square)
        
        self.tik_instance.vsub(mask, self.out_ub, self.out_ub, self.out_square, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(self.diou_gm[gm_offset], self.out_ub, 0, 1, nbust, 0, 0)

    def _rev_div(self, mask, repeat_time, x1_ub, x2_ub, y_ub):
        self.tik_instance.vrec(mask, self.div_rec_1, x1_ub, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.div_rec_2, self.div_rec_1, x1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.div_rec_2, self.div_rec_2, -1, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.div_rec_2, self.div_rec_2, 2, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.div_rec_2, self.div_rec_2, self.div_rec_1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.div_rec_1, self.div_rec_2, x1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.div_rec_1, self.div_rec_1, -1, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.div_rec_1, self.div_rec_1, 2, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.div_rec_1, self.div_rec_1, self.div_rec_2, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, y_ub, self.div_rec_1, x2_ub, repeat_time, 1, 1, 1, 8, 8, 8)


def _box_shape_check(input_name, shape):
    shape_len = len(shape)
    if shape_len != 2:
        error_detail = "the shape len should be 2"
        error_manager_vector.raise_err_input_shape_invalid("diou", input_name, error_detail)
    first_shape_dim = shape[0]
    if first_shape_dim != 4:
        error_detail = "the shape should be [4, n]"
        error_manager_vector.raise_err_input_shape_invalid("diou", input_name, error_detail)


# pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def diou(bboxes, gtboxes, overlap, trans=False, is_cross=True, mode="iou", kernel_name="diou"):
    """
    calculating data

    Parameters
    ----------
    bboxes : dict
        shape and dtype of bboxes, the coordinates of bbox
        shape must be [4, n]
    gtboxes : dict
        shape and dtype of gtboxes, the coordinates of bbox
        shape must be [4, m]
    overlap : dict
        shape and dtype of overlap
        result shape is [m, n] or [1, n]
    trans : bool
        transform from xywh to xyxy or not
    is_cross : bool
        if true: m must be equal to n, shape of overlap is [m, n]
        if false: shape of overlap is [1, n]
    mode : str
        ('iou','iof')
        iou : the output is inter_area / total_area
        iof : the output is inter_area / gtboxes_area
    kernel_name : str
        kernel name, default value is "diou"

    Returns
    -------
    None
    """
    bboxes_shape = bboxes.get("shape")
    gtboxes_shape = gtboxes.get("shape")

    # check whether mode is valid
    check_list = ("iou", "iof")
    if mode not in check_list:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "mode", "iou,iof", mode)

    _box_shape_check("bboxes", bboxes_shape)
    _box_shape_check("gtboxes", gtboxes_shape)
    bboxes_dtype = bboxes.get("dtype").lower()
    shape_util.compare_tensor_dict_key(bboxes, gtboxes, "dtype")
    check_list = ("float16", "float32")
    para_check.check_dtype(bboxes_dtype, check_list, param_name="bboxes")

    diou_obj = DIoU(bboxes, gtboxes, trans, is_cross, mode)
    res = diou_obj.run_tik(kernel_name)

    return res
