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
"""
combined_non_max_suppression
"""
import functools
from te import tik
from te import platform as tbe_platform
from te.utils import para_check
from impl.util import util_tik_comm_func
from impl.batch_multi_class_nms_topk import sort_within_ub
from impl.batch_multi_class_non_max_suppression import nms_for_single_class
from impl.batch_multi_class_non_max_suppression import tik_func_sort_with_ub
from impl.batch_multi_class_non_max_suppression import filter_score_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant
    """
    # scaling factor
    DOWN_FACTOR = 0.10
    # RPN compute 16 proposals per iteration
    RPN_PROPOSAL_NUM = 16
    # define the positive min value in fp16
    MIN_SCALAR_FP16 = 2 ** (-24)
    # define a fp16 value = 2**12
    TMP_SCALAR_FP16 = 2 ** 12


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-statements,too-many-locals
class CombinedNonMaxSuppression:
    """
    Function: use to store CombinedNonMaxSuppression base parameters
    Modify : 2021-2-19
    """
    def __init__(self,
                 boxes,
                 scores,
                 input_scalar_list,
                 score_thresh,
                 iou_thresh,
                 max_size_per_class,
                 max_total_size,
                 impl_mode):
        """
        Init CombinedNonMaxSuppression base parameters

        Returns
        -------
        None
        """
        boxes_shape = list(boxes.get("shape"))
        self.boxes_type = boxes.get("dtype")
        scores_shape = list(scores.get("shape"))
        # when input have no class dim, will extend 1 for input shape
        if len(scores_shape) == 2 and len(boxes_shape) == 3:
            self.boxes_shape = [boxes_shape[0], 1, boxes_shape[1], boxes_shape[2]]
            self.scores_shape = [scores_shape[0], 1, scores_shape[1]]
        else:
            self.boxes_shape = boxes_shape
            self.scores_shape = scores_shape
        self.input_scalar_list = input_scalar_list

        self.need_clip_window = False
        self.clip_window_shape = None

        self.need_valid_num = False
        self.valid_num_shape = None

        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh / (1 + iou_thresh)
        self.max_size_per_class = max_size_per_class
        self.max_total_size = max_total_size
        self.change_coordinate_frame = False

        para_check.check_shape(self.boxes_shape, min_rank=4, max_rank=4, param_name="boxes")
        para_check.check_shape(self.scores_shape, min_rank=3, max_rank=3, param_name="scores")
        # parsing input
        _, self.boxes_classes, _, _ = self.boxes_shape
        self.batch, self.classes, self.boxes_num = self.scores_shape
        if self.classes == self.boxes_classes and self.boxes_classes == 1:
            if self.max_size_per_class > self.max_total_size:
                self.max_size_per_class = self.max_total_size
        self.check_par()
        # whether down the boxes to avoid fp16 overflow
        self.down_flag = False
        self.is_second_nms = False
        if impl_mode == "high_precision":
            self.is_second_nms = True

        self.tik_instance = tik.Tik()
        self.aicore_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.ub_size = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []

        # calcu output shape
        self.nmsed_boxes_shape = [self.batch, 4, self.max_total_size]
        self.nmsed_scores_shape = [self.batch, self.max_total_size]
        self.nmsed_classes_shape = [self.batch, self.max_total_size]
        self.nmsed_num_shape = [self.batch, 8]

        # for topk
        self.ub_max_topk = None
        self.l1_nms_result = None
        self.l1_nms_result_zero = None
        self.workspace_proposal_gm = None
        self.workspace_second_nms_gm = None
        self.l1_score_valid = None
        self.l1_nms_area = None
        self.l1_nms_sup = None
        self.proposal_topk_k = self.ub_size // 4 // 16
        self.proposal_topk_k = min(self.proposal_topk_k, 255 * 16)
        self.topk_loop_time = 0
        self.topk_loop_tail = 0
        self.single_loop = True
        if self.boxes_num > self.proposal_topk_k:
            self.single_loop = False
            self.topk_loop_time = self.boxes_num // self.proposal_topk_k
            self.topk_loop_tail = self.boxes_num % self.proposal_topk_k
        self.topk_loop_time_reg = self.tik_instance.Scalar(dtype="int32")
        self.topk_loop_time_reg.set_as(self.topk_loop_time)
        self.topk_loop_time_tail = self.tik_instance.Scalar(dtype="int32")
        self.topk_loop_time_tail.set_as(self.topk_loop_tail)

        # whether user set_rpn_offset, mini do not support it
        self.is_need_rpn_offset = False

        # for nms function param calc
        self.max_selected_nms_num_in_ub = \
            ceil_div(max_size_per_class, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM
        # record the output nms num for one class
        self.selected_proposals_cnt = self.tik_instance.Scalar(dtype="uint16")
        # record the proposal burst num for one loop, value = 128 or self.proposal_topk_k % 128
        self.handling_proposals_cnt = self.tik_instance.Scalar(dtype="uint16")
        # init a scalar value = 0
        self.zero_scalar = self.tik_instance.Scalar(dtype="uint16")
        self.zero_scalar.set_as(0)
        # init a scalar value = 1
        self.one_scalar = self.tik_instance.Scalar(dtype="uint16")
        self.one_scalar.set_as(1)
        # init a fp16 scalar for output class
        self.nms_class_idx = self.tik_instance.Scalar(dtype="float16")
        self.nms_class_idx.set_as(0)
        # init 4 clip to windows scalar
        if self.need_clip_window:
            if self.change_coordinate_frame:
                self.down_flag = False
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype="float16")] * 6
            else:
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype="float16")] * 4
        else:
            self.clip_window_value_list = None
        # init 1 valid num scalar
        self.valid_num_value = self.tik_instance.Scalar(dtype="int32")

        self.down_scalar_list = None
        # init down scalar
        if self.down_flag:
            self.down_scalar_list = [self.tik_instance.Scalar(dtype="float16")] * 2
            self.down_scalar_list[0].set_as(Constant.DOWN_FACTOR)
            self.down_scalar_list[1].set_as(1 / Constant.DOWN_FACTOR)

    def check_par(self):
        """
        check_par
        """
        def _error_code_002_check(op_name, param_name, value_range, value):
            """
            _error_code_002_check
            """
            if value < value_range[0] or value > value_range[1]:
                error_info = {
                    'errCode': para_check.OP_ERROR_CODE_002,
                    'op_name': op_name,
                    'param_name': param_name,
                    'min_value': value_range[0],
                    'max_value': value_range[1],
                    'real_value': value
                }
                raise RuntimeError(error_info,
                                   "In op[{op_name}], the parameter[{param_name}] should be in"
                                   " the range of [{min_value}, {max_value}],"
                                   " but actually is [{real_value}].".format(**error_info))

        _error_code_002_check("CombinedNonMaxSuppression", "max_size_per_class",
                              [1, 1000], self.max_size_per_class)
        _error_code_002_check("CombinedNonMaxSuppression", "max_total_size",
                              [1, 1000], self.max_total_size)
        _error_code_002_check("CombinedNonMaxSuppression", "classes num from input scores shape",
                              [1, 200], self.classes)

        para_check.check_dtype(self.boxes_type, ("float16",), param_name="boxes")

    def get_tik_instance(self):
        """
        get_tik_instance
        """
        return self.tik_instance

    # 'pylint: disable=unused-argument
    @staticmethod
    def get_l1_core_idx(core_idx):
        """
        get l1 core idx
        """
        return 0

    def build_tik_instance(self, kernel_name_value):
        """
        build_tik_instance
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   output_files_path=None,
                                   enable_l2=False)

        return self.tik_instance

    def init_tik_mem(self):
        """
        init tik gm mem
        """
        # init gm input
        boxes_gm = self.tik_instance.Tensor("float16", self.boxes_shape, name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor("float16", self.scores_shape, name="scores_gm", scope=tik.scope_gm)

        clip_window_gm = None
        valid_num_gm = None
        if self.need_clip_window:
            clip_window_gm = self.tik_instance.Tensor("float16", self.clip_window_shape,
                                                      name="clip_window_gm", scope=tik.scope_gm)
        if self.need_valid_num:
            valid_num_gm = self.tik_instance.Tensor("int32", self.valid_num_shape,
                                                    name="valid_num_gm", scope=tik.scope_gm)
        if self.need_valid_num and self.need_clip_window:
            self.input_gm_list = [boxes_gm, scores_gm, clip_window_gm, valid_num_gm]
        elif self.need_clip_window:
            self.input_gm_list = [boxes_gm, scores_gm, clip_window_gm]
        elif self.need_valid_num:
            self.input_gm_list = [boxes_gm, scores_gm, valid_num_gm]
        else:
            self.input_gm_list = [boxes_gm, scores_gm]

        for input_scalar in self.input_scalar_list:
            scalar_dtype = input_scalar.get("dtype")
            scalar_tensor = self.tik_instance.Tensor(scalar_dtype, [1],
                                                     name="input_scalar", scope=tik.scope_gm)
            self.input_gm_list.append(scalar_tensor)

        # init gm output
        nmsed_boxes_gm = self.tik_instance.Tensor("float16", self.nmsed_boxes_shape,
                                                  name="nmsed_boxes_gm", scope=tik.scope_gm)
        nmsed_scores_gm = self.tik_instance.Tensor("float16", self.nmsed_scores_shape,
                                                   name="nmsed_scores_gm", scope=tik.scope_gm)
        nmsed_classes_gm = self.tik_instance.Tensor("float16", self.nmsed_classes_shape,
                                                    name="nmsed_classes_gm", scope=tik.scope_gm)
        nmsed_num_gm = self.tik_instance.Tensor("int32", self.nmsed_num_shape,
                                                name="nmsed_num_gm", scope=tik.scope_gm)
        self.output_gm_list = [nmsed_boxes_gm, nmsed_scores_gm, nmsed_classes_gm, nmsed_num_gm]

        # init l1 buff for save multi class nms result, size = [classes, self.max_selected_nms_num_in_ub, 8]
        self.l1_nms_result = self.tik_instance.Tensor("float16", (1, self.classes, self.max_selected_nms_num_in_ub, 8),
                                                      name="l1_nms_result", scope=tik.scope_cbuf)

        if self.is_second_nms:
            # init l1 buff for save multi class nms area, size = [self.max_selected_nms_num_in_ub]
            self.l1_nms_area = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub,),
                                                        name="l1_nms_area_tmp", scope=tik.scope_cbuf)
            # init l1 buff for save multi class nms sup, size = [self.max_selected_nms_num_in_ub]
            self.l1_nms_sup = self.tik_instance.Tensor("uint16", (self.max_selected_nms_num_in_ub,),
                                                       name="l1_nms_sup_tmp", scope=tik.scope_cbuf)

        # zero data in l1
        self.l1_nms_result_zero = \
            self.tik_instance.Tensor("float16", (1, self.max_selected_nms_num_in_ub, 8),
                                     name="l1_nms_result_zero", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            ub_nms_result = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub, 8),
                                                     name="ub_nms_result", scope=tik.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, ub_nms_result, 0,
                                               self.max_selected_nms_num_in_ub * 8)
            loop_burst_len = (self.max_selected_nms_num_in_ub * 8) // 16
            self.tik_instance.data_move(self.l1_nms_result_zero,
                                        ub_nms_result, 0, 1, loop_burst_len, 0, 0)
        # workspace
        self.workspace_proposal_gm = self.tik_instance.Tensor("float16",
                                                              [self.aicore_num,
                                                               total_num(self.l1_nms_result.shape[1:]) + 128],
                                                              name="workspace_proposal_gm",
                                                              scope=tik.scope_gm, is_workspace=True)
        # workspace for second nms
        if self.is_second_nms:
            self.workspace_second_nms_gm = self.tik_instance.Tensor("float16",
                                                                    [self.aicore_num,
                                                                     self.boxes_num * 8],
                                                                    name="workspace_second_nms_gm",
                                                                    scope=tik.scope_gm, is_workspace=True)

    def init_tik_ub_mem_for_nms(self):
        """
        init_tik_ub_mem_for_nms
        """
        ub_selected_proposals = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub, 8],
                                                         name="ub_selected_proposals", scope=tik.scope_ubuf)
        ub_selected_area = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub],
                                                    name="ub_selected_area", scope=tik.scope_ubuf)
        ub_sup_vec = self.tik_instance.Tensor("uint16", [self.max_selected_nms_num_in_ub], name="ub_sup_vec",
                                              scope=tik.scope_ubuf)

        # when is_need_rpn_offset set rpn offset for vaadd and viou
        # else x2/y2 will do vadds -1 before nms and do vadds 1 after nms
        if self.is_need_rpn_offset:
            self.tik_instance.set_rpn_offset(0.0)

        topk_out_num = self.proposal_topk_k
        if self.boxes_num < self.proposal_topk_k:
            topk_out_num = self.boxes_num
        nms_var_dict = {
            # topk_out_info mean : nms input info
            "topk_out_ub": self.ub_max_topk,
            "topk_out_num": topk_out_num,
            # selected proposal info
            "selected_proposal_ub": ub_selected_proposals,
            "selected_area_ub": ub_selected_area,
            "sup_vec_ub": ub_sup_vec,
            # scalar reg info
            "zero_scalar": self.zero_scalar,
            "one_scalar": self.one_scalar,
            "selected_proposals_cnt": self.selected_proposals_cnt,
            "handling_proposals_cnt": self.handling_proposals_cnt,
            # nms output info
            "output_num": self.max_size_per_class
        }

        return nms_var_dict

    def init_tik_ub_mem_for_topk(self):
        """
        init_tik_ub_mem_for_topk
        """
        # init one ub for topk output
        self.ub_max_topk = self.tik_instance.Tensor("float16", (self.proposal_topk_k, 8),
                                                    name="ub_max_topk", scope=tik.scope_ubuf)

    def get_core_schedule(self):
        """
        get_core_schedule
        """
        if self.max_total_size < 16:
            self.aicore_num = 1
        batch_per_core = ceil_div(self.batch, self.aicore_num)
        core_used = ceil_div(self.batch, batch_per_core)
        batch_last_core = self.batch - (core_used - 1) * batch_per_core
        self.aicore_num = core_used

        return core_used, batch_per_core, batch_last_core


def total_num(shape):
    """
    the return object is total num
    """
    shape_total_num = functools.reduce(lambda a, b: a * b, shape)
    return shape_total_num


def ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def get_class_tensor(tik_instance, class_ub, class_num, len_per_class, start_class=0.0):
    """
    get class tensor
    """
    util_tik_comm_func.tik_func_vector(tik_instance, class_ub, start_class, len_per_class)
    with tik_instance.for_range(1, class_num) as _class_idx:
        dst_offset = _class_idx * len_per_class
        src_offset = (_class_idx - 1) * len_per_class
        _repeat_time = len_per_class // 128
        _repeat_tail = len_per_class % 128
        if _repeat_time != 0:
            tik_instance.vadds(128, class_ub[dst_offset], class_ub[src_offset], 1.0,
                               _repeat_time, 1, 1, 8, 8)
            dst_offset = 128 * _repeat_time + dst_offset
            src_offset = 128 * _repeat_time + src_offset
        if _repeat_tail != 0:
            tik_instance.vadds(_repeat_tail, class_ub[dst_offset], class_ub[src_offset], 1.0,
                               1, 1, 1, 8, 8)


def copy_tail_data(tik_instance, gm_dst_info, ub_src_info, gm_workspace_info, copy_len):
    """
    copy_tail_data when output is not align, will use workspace to align force
    """
    gm_dst, gm_dst_offset = gm_dst_info
    ub_src, ub_src_offset = ub_src_info
    gm_workspace, gm_workspace_offset = gm_workspace_info
    data_type = ub_src.dtype
    if data_type in ("float32", "int32"):
        block_num = 8
    else:
        block_num = 16
    copy_nbust_len = copy_len // block_num
    copy_tail_offset = copy_len % block_num
    tik_instance.data_move(gm_dst[gm_dst_offset], ub_src[ub_src_offset], 0, 1, copy_nbust_len, 0, 0)
    tik_instance.data_move(gm_workspace[gm_workspace_offset],
                           ub_src[ub_src_offset + (copy_nbust_len - 1) * block_num],
                           0, 1, 2, 0, 0)
    tik_instance.data_move(ub_src[ub_src_offset], gm_workspace[gm_workspace_offset + copy_tail_offset],
                           0, 1, 1, 0, 0)
    tik_instance.data_move(gm_dst[gm_dst_offset + copy_tail_offset + (copy_nbust_len - 1) * block_num],
                           ub_src[ub_src_offset], 0, 1, 1, 0, 0)


def clip_boxes_compute(tik_instance, clip_ub, clip_value, clip_num, clip_flag=True):
    """
    clip_boxes with value
    """
    if not clip_flag:
        return
    with tik_instance.new_stmt_scope():
        clip_min_ub = tik_instance.Tensor(clip_ub.dtype, [16], name="clip_min_ub", scope=tik.scope_ubuf)
        clip_max_ub = tik_instance.Tensor(clip_ub.dtype, [16], name="clip_max_ub", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vector(tik_instance, clip_min_ub, clip_value[0], 16)
        util_tik_comm_func.tik_func_vector(tik_instance, clip_max_ub, clip_value[1], 16)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmax", clip_ub, clip_ub, clip_min_ub,
                                            clip_num, 1, 1, 0, 8, 8, 0)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmin", clip_ub, clip_ub, clip_max_ub,
                                            clip_num, 1, 1, 0, 8, 8, 0)


# 'pylint: disable=too-many-branches
def batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes, ub_result_boxes_class,
                                   output_batch_offset, workspace_core_offset, clip_flag=False):
    """
    batch_multi_class_nms_copy_out
    """
    clip_value = [0.0, 1.0]
    core_used = nms.aicore_num
    workspace_flag = False
    if (core_used > 1) and (nms.max_total_size % 16 != 0):
        workspace_flag = True

    workspace = nms.workspace_proposal_gm
    down_scalar = None
    if nms.down_flag:
        down_scalar = nms.down_scalar_list[1]
    loop_burst_len = ceil_div(nms.max_total_size, 16)
    apply_men_len = ceil_div(nms.max_total_size, 16)
    less_flag = False
    if nms.max_selected_nms_num_in_ub * nms.classes < nms.max_total_size:
        less_flag = True
        loop_burst_len = ceil_div(nms.max_selected_nms_num_in_ub * nms.classes, 16)
    score_thresh = nms.score_thresh
    _batch = output_batch_offset // nms.max_total_size
    ub_scores_valid_mask = tik_instance.Tensor("float16", [apply_men_len * 16],
                                               name="ub_scores_valid_mask", scope=tik.scope_ubuf)
    # process scores
    with tik_instance.new_stmt_scope():
        # scores
        ub_out_scores = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_scores", scope=tik.scope_ubuf)
        ub_out_scores_valid = tik_instance.Tensor("int32", [16], name="ub_out_scores_valid",
                                                  scope=tik.scope_ubuf)
        if less_flag:
            util_tik_comm_func.tik_func_vector(tik_instance, ub_out_scores, 0, apply_men_len * 16)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes_class, ub_out_scores, loop_burst_len, 3)
        filter_score_compute(tik_instance, ub_out_scores, ub_out_scores_valid, ub_scores_valid_mask,
                             nms.max_total_size, score_thresh)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[1][output_batch_offset], ub_out_scores,
                                   0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[1], output_batch_offset],
                           [ub_out_scores, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)

        tik_instance.data_move(nms.output_gm_list[3][_batch * 8], ub_out_scores_valid,
                               0, 1, 1, 0, 0)
        # x1
        ub_out_box_x1 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_x1", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_x1, loop_burst_len, 0)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmul", ub_out_box_x1, ub_scores_valid_mask, ub_out_box_x1,
                                            apply_men_len * 16)
        if nms.down_flag:
            util_tik_comm_func.tik_func_vmuls(tik_instance, ub_out_box_x1, ub_out_box_x1,
                                              down_scalar, nms.max_total_size)
        # y1
        ub_out_box_y1 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_y1", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_y1, loop_burst_len, 1)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmul", ub_out_box_y1, ub_scores_valid_mask, ub_out_box_y1,
                                            apply_men_len * 16)
        if nms.down_flag:
            util_tik_comm_func.tik_func_vmuls(tik_instance, ub_out_box_y1, ub_out_box_y1,
                                              down_scalar, nms.max_total_size)
        clip_boxes_compute(tik_instance, ub_out_box_x1, clip_value, nms.max_total_size, clip_flag)
        clip_boxes_compute(tik_instance, ub_out_box_y1, clip_value, nms.max_total_size, clip_flag)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4], ub_out_box_x1,
                               0, 1, apply_men_len, 0, 0)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 + nms.max_total_size],
                               ub_out_box_y1, 0, 1, apply_men_len, 0, 0)

        # x2
        ub_out_box_x2 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_x2", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_x2, loop_burst_len, 2)

        if not nms.is_need_rpn_offset:
            util_tik_comm_func.tik_func_vadds(tik_instance, ub_out_box_x2, ub_out_box_x2, 1.0, nms.max_total_size)

        if nms.down_flag:
            util_tik_comm_func.tik_func_vmuls(tik_instance, ub_out_box_x2, ub_out_box_x2,
                                              down_scalar, nms.max_total_size)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmul", ub_out_box_x2, ub_scores_valid_mask, ub_out_box_x2,
                                            apply_men_len * 16)
        clip_boxes_compute(tik_instance, ub_out_box_x2, clip_value, nms.max_total_size, clip_flag)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 + nms.max_total_size * 2],
                               ub_out_box_x2, 0, 1, apply_men_len, 0, 0)

        # y2
        ub_out_box_y2 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_y2", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_y2, loop_burst_len, 3)

        if not nms.is_need_rpn_offset:
            util_tik_comm_func.tik_func_vadds(tik_instance, ub_out_box_y2, ub_out_box_y2, 1.0, nms.max_total_size)

        if nms.down_flag:
            util_tik_comm_func.tik_func_vmuls(tik_instance, ub_out_box_y2, ub_out_box_y2,
                                              down_scalar, nms.max_total_size)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmul", ub_out_box_y2, ub_scores_valid_mask, ub_out_box_y2,
                                            apply_men_len * 16)
        clip_boxes_compute(tik_instance, ub_out_box_y2, clip_value, nms.max_total_size, clip_flag)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 + nms.max_total_size * 3],
                                   ub_out_box_y2, 0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[0], output_batch_offset * 4 + nms.max_total_size * 3],
                           [ub_out_box_y2, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)
        # class
        ub_out_class = tik_instance.Tensor("float16", [apply_men_len * 16],
                                           name="ub_out_class", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes_class, ub_out_class, loop_burst_len, 0)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[2][output_batch_offset], ub_out_class,
                                   0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[2], output_batch_offset],
                           [ub_out_class, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)


def batch_multi_class_nms_output(tik_instance, core_idx, _batch_idx, nms, clip_flag):
    """
    do batch_multi_class_nms_output

    Parameters:
    ----------
    tik_instance : tik_instance.
    _batch_idx : int.
        the process batch
    nms : class.
        all par for nms
    clip_flag: bool:
        whether clip the boxes by value (0, 1)
    Returns
    -------
    None
    """
    result_total = total_num(nms.l1_nms_result.shape[1:])
    class_num = nms.classes
    # get score batch offset
    output_batch_offset = _batch_idx * nms.max_total_size
    workspace = nms.workspace_proposal_gm
    workspace_offset = core_idx * nms.workspace_proposal_gm.shape[-1]
    if nms.classes * nms.max_selected_nms_num_in_ub < nms.proposal_topk_k:
        # when all output is less nms.proposal_topk_k
        # only use topk with ub for output proposal
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            ub_result_boxes_class = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                        name="ub_result_boxes_class", scope=tik.scope_ubuf)
            l1_buffer = nms.l1_nms_result
            l1_offset = [nms.get_l1_core_idx(core_idx), 0, 0, 0]
            loop_burst_len = result_total // 16
            tik_instance.data_move(ub_result_boxes, l1_buffer[l1_offset],
                                   0, 1, loop_burst_len, 0, 0)
            tik_instance.data_move(ub_result_boxes_class, l1_buffer[l1_offset],
                                   0, 1, loop_burst_len, 0, 0)
            with tik_instance.new_stmt_scope():
                ub_class_all = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub * nms.classes],
                                                   name="ub_class_all", scope=tik.scope_ubuf)
                get_class_tensor(tik_instance, ub_class_all, class_num, nms.max_selected_nms_num_in_ub)

                trans_repeat = ceil_div(nms.max_selected_nms_num_in_ub * nms.classes, 16)
                util_tik_comm_func.tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 1)
                tik_instance.data_move(workspace[workspace_offset], ub_result_boxes_class,
                                       0, 1, loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, workspace[workspace_offset + 1],
                                       0, 1, loop_burst_len, 0, 0)
                util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes_class,
                                                     ub_class_all, trans_repeat, 3)
                util_tik_comm_func.tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 4)

            if nms.classes != 1:
                sort_within_ub(tik_instance, ub_result_boxes_class, result_total // 8)
                sort_within_ub(tik_instance, ub_result_boxes, result_total // 8)

            with tik_instance.new_stmt_scope():
                batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes,
                                               ub_result_boxes_class, output_batch_offset,
                                               workspace_offset, clip_flag)
    else:
        l1_buffer = nms.l1_nms_result
        copy_classes_num = nms.proposal_topk_k // nms.max_selected_nms_num_in_ub // 2
        copy_loop = nms.classes // copy_classes_num
        copy_tail = nms.classes % copy_classes_num
        tmp_output_proposal_num = ceil_div(nms.max_total_size, 16) * 16
        ub_out_result = tik_instance.Tensor("float16", [tmp_output_proposal_num, 8],
                                            name="ub_out_result", scope=tik.scope_ubuf)
        ub_out_result_class = tik_instance.Tensor("float16", [tmp_output_proposal_num, 8],
                                                  name="ub_out_result_class", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vector(tik_instance, ub_out_result, 0.0, tmp_output_proposal_num * 8)
        util_tik_comm_func.tik_func_vector(tik_instance, ub_out_result_class, 0.0, tmp_output_proposal_num * 8)
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [copy_classes_num * nms.max_selected_nms_num_in_ub, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            ub_result_boxes_class = tik_instance.Tensor("float16", [copy_classes_num * nms.max_selected_nms_num_in_ub,
                                                                    8],
                                                        name="ub_result_boxes_class", scope=tik.scope_ubuf)
            ub_class_all = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub * copy_classes_num],
                                               name="ub_class_all", scope=tik.scope_ubuf)
            get_class_tensor(tik_instance, ub_class_all, copy_classes_num,
                             nms.max_selected_nms_num_in_ub, copy_classes_num * -1)

            def _do_copy_and_vconcat_class(_l1_offset, _loop_burst_len):
                tik_instance.data_move(ub_result_boxes, l1_buffer[_l1_offset],
                                       0, 1, _loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, l1_buffer[_l1_offset],
                                       0, 1, _loop_burst_len, 0, 0)
                # get copy_classes_num sort
                util_tik_comm_func.tik_func_vadds(tik_instance, ub_class_all, ub_class_all, copy_classes_num * 1.0,
                                                  nms.max_selected_nms_num_in_ub * copy_classes_num)
                _trans_repeat = ceil_div(nms.max_selected_nms_num_in_ub * copy_classes_num, 16)
                util_tik_comm_func.tik_func_vconcat(tik_instance, ub_result_boxes_class,
                                                    ub_class_all, _trans_repeat, 1)
                tik_instance.data_move(workspace[workspace_offset], ub_result_boxes_class,
                                       0, 1, _loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, workspace[workspace_offset + 1],
                                       0, 1, _loop_burst_len, 0, 0)
                with tik_instance.new_stmt_scope():
                    ub_class_tmp = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub * copy_classes_num],
                                                       name="ub_class_tmp", scope=tik.scope_ubuf)
                    util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes_class,
                                                         ub_class_tmp, _trans_repeat, 3)
                    util_tik_comm_func.tik_func_vconcat(tik_instance, ub_result_boxes_class,
                                                        ub_class_tmp, _trans_repeat, 4)

            with tik_instance.for_range(0, copy_loop) as _class_idx:
                l1_offset = [nms.get_l1_core_idx(core_idx), _class_idx * copy_classes_num, 0, 0]
                loop_burst_len = copy_classes_num * nms.max_selected_nms_num_in_ub * 8 // 16
                _do_copy_and_vconcat_class(l1_offset, loop_burst_len)
                sort_within_ub(tik_instance, ub_result_boxes, copy_classes_num * nms.max_selected_nms_num_in_ub)
                sort_within_ub(tik_instance, ub_result_boxes_class, copy_classes_num * nms.max_selected_nms_num_in_ub)
                tik_func_sort_with_ub(tik_instance, [ub_out_result, ub_result_boxes],
                                      [ub_out_result, ub_result_boxes], tmp_output_proposal_num)
                tik_func_sort_with_ub(tik_instance, [ub_out_result_class, ub_result_boxes_class],
                                      [ub_out_result_class, ub_result_boxes_class], tmp_output_proposal_num)

            if copy_tail != 0:
                l1_offset = [nms.get_l1_core_idx(core_idx), copy_loop * copy_classes_num, 0, 0]
                loop_burst_len = copy_tail * nms.max_selected_nms_num_in_ub * 8 // 16
                _do_copy_and_vconcat_class(l1_offset, loop_burst_len)
                sort_within_ub(tik_instance, ub_result_boxes, copy_tail * nms.max_selected_nms_num_in_ub)
                sort_within_ub(tik_instance, ub_result_boxes_class, copy_tail * nms.max_selected_nms_num_in_ub)
                if copy_tail * nms.max_selected_nms_num_in_ub < tmp_output_proposal_num:
                    dup_len = tmp_output_proposal_num - copy_tail * nms.max_selected_nms_num_in_ub
                    dup_offset = copy_tail * nms.max_selected_nms_num_in_ub
                    util_tik_comm_func.tik_func_vector(tik_instance, ub_result_boxes[dup_offset:], 0.0, dup_len * 8)
                    util_tik_comm_func.tik_func_vector(tik_instance, ub_result_boxes_class[dup_offset:],
                                                       0.0, dup_len * 8)
                tik_func_sort_with_ub(tik_instance, [ub_out_result, ub_result_boxes],
                                      [ub_out_result, ub_result_boxes], tmp_output_proposal_num)
                tik_func_sort_with_ub(tik_instance, [ub_out_result_class, ub_result_boxes_class],
                                      [ub_out_result_class, ub_result_boxes_class], tmp_output_proposal_num)
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_copy_out(tik_instance, nms, ub_out_result, ub_out_result_class,
                                           output_batch_offset, workspace_offset, clip_flag)


# 'pylint: disable=unused-argument
def check_supported(boxes, scores, max_output_size_per_class,
                    max_total_size, iou_threshold, score_threshold,
                    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections,
                    pad_per_class, clip_boxes,
                    kernel_name="combined_non_max_suppression"):
    """
    check_supported: check whether the aicore support this case

    if the valid_detections_shape shape len = 2, do in aicore
    """
    valid_detections_shape = valid_detections.get("ori_shape")

    if len(valid_detections_shape) == 2:
        return True, ""
    reason = "if the valid_detections_shape shape len != 2, not supported by aicore"
    return False, reason


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def combined_non_max_suppression(boxes, scores, max_output_size_per_class,
                                 max_total_size, iou_threshold, score_threshold,
                                 nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections,
                                 pad_per_class, clip_boxes,
                                 kernel_name="combined_non_max_suppression",
                                 impl_mode="high_performance"):
    """
    do non_max_suppression for multi batch and multi class
    step 1- clip boxes use clip_window, when the area of boxes after clip, change the score = 0
    step 2- filter score, when the score is less score_threshold, change the score = 0
    step 3- filter valid num use num_valid_boxes
    step 4- trans the box and score to proposal
    step 5- sort the input proposals and get 4094 proposals
    step 6- do nms for each class in each batch use top 4094 proposals
    step 7- concat all class nms result in each batch
    step 8- sort the proposals and output the max_total_size box/class/score

    Parameters:
    ----------
    boxes : dict.
        shape, dtype of boxes, a 4D Tensor of type float16 with shape (batch, num_anchors, num_classes, 4).
        "batch" indicates the batch size of image,
        and "num_anchors" indicates num of boxes, and "num_classes" indicates classes of detect.
        and the value "4" refers to "x0", "x1", "y0", and "y1".
    scores : dict.
        shape, dtype of scores
        a 3D Tensor of type float16 with shape (batch, num_anchors, num_classes).
    max_output_size_per_class : dict.
        A required scalar of type int, specifying the nms output num per class.
    max_total_size : dict.
        A required scalar of type int, specifying the the nms output num per batch.
    iou_threshold : dict.
        A required scalar of type float32, specifying the nms iou iou_threshold
    score_threshold : dict.
        A required scalar of type float32, specifying the score filter iou iou_threshold.
    nmsed_boxes : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size, 4).
        specifying the output nms boxes per batch
    nmsed_scores : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size).
        specifying the output nms score per batch
    nmsed_classes : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size).
        specifying the output nms class per batch
    valid_detections : dict.
        A 1D Tensor of type int32 with shape (batch,),
        specifying the valid num of nmsed_boxes
    pad_per_class : bool.
        A required attribute of type bool, whether to pad result to max_total_size.
    clip_boxes : bool.
        A required attribute of type bool, whether clip the output boxes by [0, 1]
    kernel_name : str.
        cce kernel name, default value is "combined_non_max_suppression"
    impl_mode: str.
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    tik_instance
    """
    nms = CombinedNonMaxSuppression(boxes, scores,
                                    [max_output_size_per_class, max_total_size,
                                     iou_threshold, score_threshold],
                                    score_threshold.get("const_value")[0],
                                    iou_threshold.get("const_value")[0],
                                    max_output_size_per_class.get("const_value")[0],
                                    max_total_size.get("const_value")[0], impl_mode)
    # init ub
    core_used, batch_per_core, batch_last_core = nms.get_core_schedule()
    class_num = nms.classes
    nms.init_tik_mem()
    tik_instance = nms.get_tik_instance()

    def _run_one_core(_real_batch_idx, _real_core_idx):
        with tik_instance.for_range(0, class_num) as _class_idx:
            nms.selected_proposals_cnt.set_as(0)
            with tik_instance.new_stmt_scope():
                nms_for_single_class(_real_batch_idx, _class_idx, nms, _real_core_idx)

        # process all class output result is in l1_nms_result, will process output
        # step 1 sort all select proposal with boxes
        # step 2 sort all select proposal with classes score
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_output(tik_instance, _real_core_idx, _real_batch_idx, nms, clip_boxes)

    # do nms with multi cores
    with tik_instance.for_range(0, core_used, block_num=core_used) as _core_idx:
        if batch_per_core == batch_last_core or core_used == 1:
            with tik_instance.for_range(0, batch_per_core) as _batch_idx:
                real_batch_idx = _core_idx * batch_per_core + _batch_idx
                _run_one_core(real_batch_idx, _core_idx)
        else:
            with tik_instance.if_scope(_core_idx < core_used - 1):
                with tik_instance.for_range(0, batch_per_core) as _batch_idx:
                    real_batch_idx = _core_idx * batch_per_core + _batch_idx
                    _run_one_core(real_batch_idx, _core_idx)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, batch_last_core) as _batch_idx:
                    real_batch_idx = _core_idx * batch_per_core + _batch_idx
                    _run_one_core(real_batch_idx, _core_idx)

    return nms.build_tik_instance(kernel_name)
