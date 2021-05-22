# Copyright 2019 Huawei Technologies Co., Ltd
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
nms_with_mask
"""

import te.platform as tbe_platform
from te import tik
from te.utils import para_check
from te.utils.error_manager import error_manager_vector

# shape's dim of input must be 2
INPUT_DIM = 2

# scaling factor
DOWN_FACTOR = 0.054395

# process 128 proposals at a time
BURST_PROPOSAL_NUM = 128

# RPN compute 16 proposals per iteration
RPN_PROPOSAL_NUM = 16

# the coordinate column contains x1,y1,x2,y2
COORD_COLUMN_NUM = 4

# valid proposal column contains x1,y1,x2,y2,score
VALID_COLUMN_NUM = 5

# each region proposal contains eight elements
ELEMENT_NUM = 8

CONFIG_DATA_ALIGN = 32

REPEAT_TIMES_MAX = 255

# 7967 is [1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0] for 16 inputs
PATTERN_VALUE_7967 = 7967

# [1 1 1 1 1 0 0 0   1 1 1 1 1 0 0 0   1 1 1 1 1 0 0 0   1 1 1 1 1 0 0 0] one uint32 can handle selection of 32 elems
PATTERN_VALUE_522133279 = 522133279

# 2 ** 0 + 2 ** 8
PATTERN_VALUE_FP16_X1 = 257

# 2 ** 1 + 2 ** 9
PATTERN_VALUE_FP16_Y1 = 514

# 2 ** 2 + 2 ** 10
PATTERN_VALUE_FP16_X2 = 1028

# 2 ** 3 + 2 ** 11
PATTERN_VALUE_FP16_Y2 = 2056

# 2 ** 0 + 2 ** 8 + 2 ** 16 + 2 ** 24
PATTERN_VALUE_FP32_X1 = 16843009

# 2 ** 1 + 2 ** 9 + 2 ** 17 + 2 ** 25
PATTERN_VALUE_FP32_Y1 = 33686018

# 2 ** 2 + 2 ** 10 + 2 ** 18 + 2 ** 26
PATTERN_VALUE_FP32_X2 = 67372036

# 2 ** 3 + 2 ** 11 + 2 ** 19 + 2 ** 27
PATTERN_VALUE_FP32_Y2 = 134744072


def _ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def _ceiling(value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value and can divide factor
    """
    result = (value + (factor - 1)) // factor * factor
    return result


# pylint: disable=invalid-name
def _get_src_tensor(ib):
    """
    Produce two tensors with all zero or all one

    Parameters
    ----------
    ib: TIK API

    Returns
    -------
    src0_ub: the tensor with zero
    src1_ub: the tensor with one
    """
    one_scalar = ib.Scalar(dtype="float16", name="one_scalar", init_value=1.0)
    zero_scalar = ib.Scalar(dtype="float16", name="zero_scalar", init_value=0.0)
    src0_ub = ib.Tensor("float16", (BURST_PROPOSAL_NUM,), name="src0_ub", scope=tik.scope_ubuf)
    src1_ub = ib.Tensor("float16", (BURST_PROPOSAL_NUM,), name="src1_ub", scope=tik.scope_ubuf)
    ib.vector_dup(128, src0_ub, zero_scalar, 1, 1, 8)
    ib.vector_dup(128, src1_ub, one_scalar, 1, 1, 8)

    return src0_ub, src1_ub


# pylint: disable=invalid-name
def _get_reduced_proposal(ib, out_proposal, output_proposals_final, in_proposal, coord_addr):
    """
    Reduce input proposal when input boxes out of range.

    Parameters
    ----------
    ib: TIK API

    out_proposal: output proposal after reduce

    output_proposals_final: output proposal with boxes and scores, support [128,5]

    in_proposal: input proposal with boxes and scores, support [128,8]

    coord_addr: intermediate proposal after reshape

    Returns
    -------
    None
    """
    # extract original coordinates
    if tbe_platform.api_check_support("tik.vreduce", "float16") and tbe_platform.api_check_support(
            "tik.v4dtrans", "float16"):
        with ib.for_range(0, VALID_COLUMN_NUM) as i:
            ib.vextract(coord_addr[BURST_PROPOSAL_NUM * i], in_proposal, BURST_PROPOSAL_NUM // RPN_PROPOSAL_NUM, i)
        # transpose 5*burst_proposal_num to burst_proposal_num*5, output boxes and scores
        ib.v4dtrans(True, output_proposals_final, coord_addr, BURST_PROPOSAL_NUM, VALID_COLUMN_NUM)
    else:
        with ib.for_range(0, COORD_COLUMN_NUM) as i:
            ib.vextract(coord_addr[BURST_PROPOSAL_NUM * i], in_proposal, BURST_PROPOSAL_NUM // RPN_PROPOSAL_NUM, i)

    # coordinate multiplied by down_factor to prevent out of range
    ib.vmuls(128, coord_addr, coord_addr, DOWN_FACTOR, 4, 1, 1, 8, 8)

    # add 1 for x1 and y1 because nms operate would reduces 1
    ib.vadds(128, coord_addr[0], coord_addr[0], 1.0, 1, 1, 1, 8, 8)
    ib.vadds(128, coord_addr[BURST_PROPOSAL_NUM * 1], coord_addr[BURST_PROPOSAL_NUM * 1], 1.0, 1, 1, 1, 8, 8)

    # compose new proposals
    with ib.for_range(0, COORD_COLUMN_NUM) as i:
        ib.vconcat(out_proposal, coord_addr[BURST_PROPOSAL_NUM * i], BURST_PROPOSAL_NUM // RPN_PROPOSAL_NUM, i)


# pylint: disable=too-many-locals,too-many-arguments
def _tik_func_nms_single_core_multithread(input_shape, thresh, total_output_proposal_num, kernel_name_var):
    """
    Compute output boxes after non-maximum suppression.

    Parameters
    ----------
    input_shape: dict
        shape of input boxes, including proposal boxes and corresponding confidence scores

    thresh: float
        iou threshold

    total_output_proposal_num: int
        the number of output proposal boxes

    kernel_name_var: str
        cce kernel name

    Returns
    -------
    tik_instance: TIK API
    """
    tik_instance = tik.Tik()
    total_input_proposal_num, _ = input_shape
    proposals = tik_instance.Tensor("float16", (total_input_proposal_num, ELEMENT_NUM),
                                    name="in_proposals",
                                    scope=tik.scope_gm)
    support_vreduce = tbe_platform.api_check_support("tik.vreduce", "float16")
    support_v4dtrans = tbe_platform.api_check_support("tik.v4dtrans", "float16")
    # output shape is [N,5]
    input_ceil = _ceil_div(total_input_proposal_num * VALID_COLUMN_NUM, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM
    ret = tik_instance.Tensor("float16", (_ceil_div(input_ceil, VALID_COLUMN_NUM), VALID_COLUMN_NUM),
                              name="out_proposals",
                              scope=tik.scope_gm)

    # address is 32B aligned
    out_index = tik_instance.Tensor("int32", (_ceil_div(total_output_proposal_num, ELEMENT_NUM) * ELEMENT_NUM,),
                                    name="out_index",
                                    scope=tik.scope_gm)
    out_mask = tik_instance.Tensor("uint8",
                                   (_ceil_div(total_output_proposal_num, CONFIG_DATA_ALIGN) * CONFIG_DATA_ALIGN,),
                                   name="out_mask",
                                   scope=tik.scope_gm)
    output_index_ub = tik_instance.Tensor("int32", (BURST_PROPOSAL_NUM,), name="output_index_ub", scope=tik.scope_ubuf)
    output_mask_ub = tik_instance.Tensor("uint8", (BURST_PROPOSAL_NUM,), name="output_mask_ub", scope=tik.scope_ubuf)
    output_proposals_ub = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM, VALID_COLUMN_NUM),
                                              name="output_proposals_ub",
                                              scope=tik.scope_ubuf)

    # init tensor every 128 proposals
    fresh_proposals_ub = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM, ELEMENT_NUM),
                                             name="fresh_proposals_ub",
                                             scope=tik.scope_ubuf)
    temp_reduced_proposals_ub = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM, ELEMENT_NUM),
                                                    name="temp_reduced_proposals_ub",
                                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, temp_reduced_proposals_ub[0], 0, 8, 1, 8)

    # init middle selected proposals
    selected_reduced_proposals_ub = tik_instance.Tensor(
        "float16", (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM), ELEMENT_NUM),
        name="selected_reduced_proposals_ub",
        scope=tik.scope_ubuf)
    # init middle selected area
    selected_area_ub = tik_instance.Tensor("float16", (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM),),
                                           name="selected_area_ub",
                                           scope=tik.scope_ubuf)
    # init middle sup_vec
    sup_vec_ub = tik_instance.Tensor("uint16", (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM),),
                                     name="sup_vec_ub",
                                     scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, sup_vec_ub[0], 1, 1, 1, 8)

    # init nms tensor
    temp_area_ub = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM,), name="temp_area_ub", scope=tik.scope_ubuf)
    temp_iou_ub = tik_instance.Tensor("float16",
                                      (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM), RPN_PROPOSAL_NUM),
                                      name="temp_iou_ub",
                                      scope=tik.scope_ubuf)
    temp_join_ub = tik_instance.Tensor("float16",
                                       (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM), RPN_PROPOSAL_NUM),
                                       name="temp_join_ub",
                                       scope=tik.scope_ubuf)
    temp_sup_matrix_ub = tik_instance.Tensor("uint16", (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM),),
                                             name="temp_sup_matrix_ub",
                                             scope=tik.scope_ubuf)
    temp_sup_vec_ub = tik_instance.Tensor("uint16", (BURST_PROPOSAL_NUM,),
                                          name="temp_sup_vec_ub",
                                          scope=tik.scope_ubuf)

    if support_vreduce and support_v4dtrans:
        output_mask_f16 = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM,),
                                              name="output_mask_f16",
                                              scope=tik.scope_ubuf)
        data_zero, data_one = _get_src_tensor(tik_instance)

        middle_reduced_proposals = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM, ELEMENT_NUM),
                                                       name="middle_reduced_proposals",
                                                       scope=tik.scope_ubuf)

        # init v200 reduce param
        nms_tensor_pattern = tik_instance.Tensor(dtype="uint16",
                                                 shape=(ELEMENT_NUM,),
                                                 name="nms_tensor_pattern",
                                                 scope=tik.scope_ubuf)
        # init ori coord
        coord_addr = tik_instance.Tensor("float16", (VALID_COLUMN_NUM, BURST_PROPOSAL_NUM),
                                         name="coord_addr",
                                         scope=tik.scope_ubuf)
        # init reduce zoom coord
        zoom_coord_reduce = tik_instance.Tensor("float16", (COORD_COLUMN_NUM, BURST_PROPOSAL_NUM),
                                                name="zoom_coord_reduce",
                                                scope=tik.scope_ubuf)
        # init reduce num
        num_nms = tik_instance.Scalar(dtype="uint32")
    else:
        # init ori coord
        coord_addr = tik_instance.Tensor("float16", (COORD_COLUMN_NUM, BURST_PROPOSAL_NUM),
                                         name="coord_addr",
                                         scope=tik.scope_ubuf)
        mask = tik_instance.Scalar(dtype="uint8")

    # variables
    selected_proposals_cnt = tik_instance.Scalar(dtype="uint16")
    selected_proposals_cnt.set_as(0)
    handling_proposals_cnt = tik_instance.Scalar(dtype="uint16")
    handling_proposals_cnt.set_as(0)
    left_proposal_cnt = tik_instance.Scalar(dtype="uint16")
    left_proposal_cnt.set_as(total_input_proposal_num)
    scalar_zero = tik_instance.Scalar(dtype="uint16")
    scalar_zero.set_as(0)
    sup_vec_ub[0].set_as(scalar_zero)

    # handle 128 proposals every time
    with tik_instance.for_range(0, _ceil_div(total_input_proposal_num, BURST_PROPOSAL_NUM),
                                thread_num=1) as burst_index:
        # update counter
        with tik_instance.if_scope(left_proposal_cnt < BURST_PROPOSAL_NUM):
            handling_proposals_cnt.set_as(left_proposal_cnt)
        with tik_instance.else_scope():
            handling_proposals_cnt.set_as(BURST_PROPOSAL_NUM)

        tik_instance.data_move(fresh_proposals_ub[0], proposals[burst_index * BURST_PROPOSAL_NUM * ELEMENT_NUM], 0, 1,
                               _ceil_div(handling_proposals_cnt * RPN_PROPOSAL_NUM, CONFIG_DATA_ALIGN), 0, 0, 0)
        # reduce fresh proposal
        _get_reduced_proposal(tik_instance, temp_reduced_proposals_ub, output_proposals_ub, fresh_proposals_ub,
                              coord_addr)
        # calculate the area of reduced-proposal
        tik_instance.vrpac(temp_area_ub[0], temp_reduced_proposals_ub[0],
                           _ceil_div(handling_proposals_cnt, RPN_PROPOSAL_NUM))
        # start to update iou and or area from the first 16 proposal and get suppression vector 16 by 16 proposal
        length = tik_instance.Scalar(dtype="uint16")
        length.set_as(_ceiling(selected_proposals_cnt, RPN_PROPOSAL_NUM))
        # clear temp_sup_vec_ub
        tik_instance.vector_dup(128, temp_sup_vec_ub[0], 1, temp_sup_vec_ub.shape[0] // BURST_PROPOSAL_NUM, 1, 8)

        with tik_instance.for_range(0, _ceil_div(handling_proposals_cnt, RPN_PROPOSAL_NUM)) as i:
            length.set_as(length + RPN_PROPOSAL_NUM)
            # calculate intersection of tempReducedProposals and selReducedProposals
            tik_instance.viou(temp_iou_ub, selected_reduced_proposals_ub,
                              temp_reduced_proposals_ub[i * RPN_PROPOSAL_NUM, 0],
                              _ceil_div(selected_proposals_cnt, RPN_PROPOSAL_NUM))
            # calculate intersection of tempReducedProposals and tempReducedProposals(include itself)
            tik_instance.viou(temp_iou_ub[_ceiling(selected_proposals_cnt, RPN_PROPOSAL_NUM), 0],
                              temp_reduced_proposals_ub, temp_reduced_proposals_ub[i * RPN_PROPOSAL_NUM, 0], i + 1)
            # calculate join of tempReducedProposals and selReducedProposals
            tik_instance.vaadd(temp_join_ub, selected_area_ub, temp_area_ub[i * RPN_PROPOSAL_NUM],
                               _ceil_div(selected_proposals_cnt, RPN_PROPOSAL_NUM))
            # calculate intersection of tempReducedProposals and tempReducedProposals(include itself)
            tik_instance.vaadd(temp_join_ub[_ceiling(selected_proposals_cnt, RPN_PROPOSAL_NUM), 0], temp_area_ub,
                               temp_area_ub[i * RPN_PROPOSAL_NUM], i + 1)
            # calculate join*(thresh/(1+thresh))
            tik_instance.vmuls(128, temp_join_ub, temp_join_ub, thresh, _ceil_div(length, ELEMENT_NUM), 1, 1, 8, 8)
            # compare and generate suppression matrix
            tik_instance.vcmpv_gt(temp_sup_matrix_ub, temp_iou_ub, temp_join_ub, _ceil_div(length, ELEMENT_NUM), 1, 1,
                                  8, 8)
            # generate suppression vector
            rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)
            # non-diagonal
            rpn_cor_ir = tik_instance.rpn_cor(temp_sup_matrix_ub[0], sup_vec_ub[0], 1, 1,
                                              _ceil_div(selected_proposals_cnt, RPN_PROPOSAL_NUM))
            with tik_instance.if_scope(i > 0):
                rpn_cor_ir = tik_instance.rpn_cor(
                    temp_sup_matrix_ub[_ceiling(selected_proposals_cnt, RPN_PROPOSAL_NUM)], temp_sup_vec_ub, 1, 1, i)
            # diagonal
            tik_instance.rpn_cor_diag(temp_sup_vec_ub[i * RPN_PROPOSAL_NUM],
                                      temp_sup_matrix_ub[length - RPN_PROPOSAL_NUM], rpn_cor_ir)

        if support_vreduce and support_v4dtrans:
            with tik_instance.for_range(0, handling_proposals_cnt) as i:
                output_index_ub[i].set_as(i + burst_index * BURST_PROPOSAL_NUM)

            # get the mask tensor of temp_sup_vec_ub
            temp_tensor = temp_sup_vec_ub.reinterpret_cast_to("float16")
            cmpmask = tik_instance.vcmp_eq(128, temp_tensor, data_zero, 1, 1)
            tik_instance.mov_cmpmask_to_tensor(nms_tensor_pattern.reinterpret_cast_to("uint16"), cmpmask)

            # save the area corresponding to these filtered proposals for the next nms use
            tik_instance.vreduce(128, selected_area_ub[selected_proposals_cnt], temp_area_ub, nms_tensor_pattern, 1, 1,
                                 8, 0, 0, num_nms, "counter")
            # sup_vec_ub set as 0
            tik_instance.vector_dup(16, sup_vec_ub[selected_proposals_cnt], 0, _ceil_div(num_nms, RPN_PROPOSAL_NUM), 1,
                                    1)

            # save the filtered proposal for next nms use
            tik_instance.vector_dup(128, zoom_coord_reduce, 0, 4, 1, 8)
            tik_instance.vector_dup(128, middle_reduced_proposals, 0, 8, 1, 8)
            with tik_instance.for_range(0, COORD_COLUMN_NUM) as i:
                tik_instance.vreduce(128, zoom_coord_reduce[i, 0], coord_addr[i, 0], nms_tensor_pattern, 1, 1, 8, 0, 0,
                                     None, "counter")
            with tik_instance.for_range(0, COORD_COLUMN_NUM) as i:
                tik_instance.vconcat(middle_reduced_proposals, zoom_coord_reduce[i, 0],
                                     _ceil_div(num_nms, RPN_PROPOSAL_NUM), i)
            tik_instance.data_move(selected_reduced_proposals_ub[selected_proposals_cnt, 0], middle_reduced_proposals,
                                   0, 1, _ceil_div(num_nms * ELEMENT_NUM, RPN_PROPOSAL_NUM), 0, 0)

            selected_proposals_cnt.set_as(selected_proposals_cnt + num_nms)

            # convert the output mask from binary to decimal
            tik_instance.vsel(128, 0, output_mask_f16, cmpmask, data_one, data_zero, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vec_conv(128, "none", output_mask_ub, output_mask_f16, 1, 8, 8)
        else:
            with tik_instance.for_range(0, handling_proposals_cnt) as i:
                with tik_instance.for_range(0, VALID_COLUMN_NUM) as j:
                    # update selOriginalProposals_ub
                    output_proposals_ub[i, j].set_as(fresh_proposals_ub[i, j])
                output_index_ub[i].set_as(i + burst_index * BURST_PROPOSAL_NUM)
                with tik_instance.if_scope(temp_sup_vec_ub[i] == 0):
                    with tik_instance.for_range(0, ELEMENT_NUM) as j:
                        # update selected_reduced_proposals_ub
                        selected_reduced_proposals_ub[selected_proposals_cnt, j].set_as(temp_reduced_proposals_ub[i, j])
                    # update selected_area_ub
                    selected_area_ub[selected_proposals_cnt].set_as(temp_area_ub[i])
                    # update sup_vec_ub
                    sup_vec_ub[selected_proposals_cnt].set_as(scalar_zero)
                    mask.set_as(1)
                    output_mask_ub[i].set_as(mask)
                    # update counter
                    selected_proposals_cnt.set_as(selected_proposals_cnt + 1)
                with tik_instance.else_scope():
                    mask.set_as(0)
                    output_mask_ub[i].set_as(mask)

        left_proposal_cnt.set_as(left_proposal_cnt - handling_proposals_cnt)
        # mov target proposals to out - mte3
        tik_instance.data_move(ret[burst_index * BURST_PROPOSAL_NUM, 0], output_proposals_ub, 0, 1,
                               _ceil_div(handling_proposals_cnt * VALID_COLUMN_NUM, RPN_PROPOSAL_NUM), 0, 0, 0)
        tik_instance.data_move(out_index[burst_index * BURST_PROPOSAL_NUM], output_index_ub, 0, 1,
                               _ceil_div(handling_proposals_cnt, ELEMENT_NUM), 0, 0, 0)
        tik_instance.data_move(out_mask[burst_index * BURST_PROPOSAL_NUM], output_mask_ub, 0, 1,
                               _ceil_div(handling_proposals_cnt, CONFIG_DATA_ALIGN), 0, 0, 0)
    tik_instance.BuildCCE(kernel_name=kernel_name_var,
                          inputs=[proposals],
                          outputs=[ret, out_index, out_mask],
                          output_files_path=None,
                          enable_l2=False)
    return tik_instance


# pylint: disable=unused-argument,too-many-locals,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def nms_with_mask(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr, kernel_name="nms_with_mask"):
    """
    algorithm: nms_with_mask

    find the best target bounding box and eliminate redundant bounding boxes

    Parameters
    ----------
    box_scores: dict
        2-D shape and dtype of input tensor, only support [N, 8]
        including proposal boxes and corresponding confidence scores

    selected_boxes: dict
        2-D shape and dtype of output boxes tensor, only support [N,5]
        including proposal boxes and corresponding confidence scores

    selected_idx: dict
        the index of input proposal boxes

    selected_mask: dict
        the symbol judging whether the output proposal boxes is valid

    iou_thr: float
        iou threshold

    kernel_name: str
        cce kernel name, default value is "nms_with_mask"

    Returns
    -------
    None
    """
    # 1981 branch
    if tbe_platform.api_check_support("tik.vreduce", "float16") and not tbe_platform.api_check_support("tik.vaadd",
                                                                                                       "float16"):
        return _nms_with_mask_1981(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr, kernel_name)

    input_shape = box_scores.get("shape")
    input_dtype = box_scores.get("dtype").lower()

    # check dtype
    check_list = ("float16")
    para_check.check_dtype(input_dtype, check_list, param_name="box_scores")
    # check shape
    para_check.check_shape(input_shape, min_rank=INPUT_DIM, max_rank=INPUT_DIM, param_name="box_scores")

    support_vreduce = tbe_platform.api_check_support("tik.vreduce", "float16")
    support_v4dtrans = tbe_platform.api_check_support("tik.v4dtrans", "float16")

    # Considering the memory space of Unified_Buffer
    fp16_size = tbe_platform.get_bit_len("float16") // 8
    int32_size = tbe_platform.get_bit_len("int32") // 8
    uint8_size = tbe_platform.get_bit_len("uint8") // 8
    uint16_size = tbe_platform.get_bit_len("uint16") // 8
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # output shape is [N,5], including x1,y1,x2,y2,scores
    burst_size = BURST_PROPOSAL_NUM * int32_size + BURST_PROPOSAL_NUM * uint8_size + \
                 BURST_PROPOSAL_NUM * VALID_COLUMN_NUM * fp16_size
    # compute shape is [N,8]
    selected_size = _ceiling(input_shape[0], RPN_PROPOSAL_NUM) * ELEMENT_NUM * fp16_size + _ceiling(
        input_shape[0], RPN_PROPOSAL_NUM) * fp16_size + _ceiling(input_shape[0], RPN_PROPOSAL_NUM) * uint16_size
    # intermediate calculation results
    temp_iou_size = _ceiling(input_shape[0], RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM * fp16_size
    temp_join_size = _ceiling(input_shape[0], RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM * fp16_size
    temp_sup_matrix_size = _ceiling(input_shape[0], RPN_PROPOSAL_NUM) * uint16_size
    temp_sup_vec_size = BURST_PROPOSAL_NUM * uint16_size
    temp_area_size = BURST_PROPOSAL_NUM * fp16_size
    temp_reduced_proposals_size = BURST_PROPOSAL_NUM * ELEMENT_NUM * fp16_size
    temp_size = temp_iou_size + temp_join_size + temp_sup_matrix_size + temp_sup_vec_size + \
                temp_area_size + temp_reduced_proposals_size
    # input shape is [N,8]
    fresh_size = BURST_PROPOSAL_NUM * ELEMENT_NUM * fp16_size
    if support_vreduce and support_v4dtrans:
        coord_size = BURST_PROPOSAL_NUM * VALID_COLUMN_NUM * fp16_size
        middle_reduced_proposals_size = BURST_PROPOSAL_NUM * ELEMENT_NUM * fp16_size
        src_tensor_size = BURST_PROPOSAL_NUM * fp16_size + BURST_PROPOSAL_NUM * fp16_size
        output_mask_f16_size = BURST_PROPOSAL_NUM * fp16_size
        nms_tensor_pattern_size = ELEMENT_NUM * uint16_size
        zoom_coord_reduce = BURST_PROPOSAL_NUM * COORD_COLUMN_NUM * fp16_size
        v200_size = output_mask_f16_size + src_tensor_size + middle_reduced_proposals_size + \
                    nms_tensor_pattern_size + zoom_coord_reduce
        used_size = burst_size + selected_size + temp_size + fresh_size + coord_size + v200_size
    else:
        coord_size = BURST_PROPOSAL_NUM * COORD_COLUMN_NUM * fp16_size
        used_size = burst_size + selected_size + temp_size + fresh_size + coord_size

    if used_size > ub_size_bytes:
        error_manager_vector.raise_err_check_params_rules(
            kernel_name, "the number of input boxes out of range(%d B)" % ub_size_bytes, "used size", used_size)

    if input_shape[1] != ELEMENT_NUM:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the 2nd-dim of input boxes must be equal to 8",
                                                          "box_scores.shape", input_shape)

    output_size, _ = input_shape
    iou_thr = iou_thr / (1 + iou_thr)
    return _tik_func_nms_single_core_multithread(input_shape, iou_thr, output_size, kernel_name)


class _NMSHelper(object):
    """
    handle all input proposals, e.g. N may > 128

    idea:
                        sn's mask: sn+1     andMask(means: which idx still exists in dst), vand or vmul
        init:           [1 1 1 1 1 1 1 1]   [1 1 1 1 1 1 1 1]  init state, from 0.elem, now 0.elem
        s0's result:    [0 0 1 0 1 1 0 1]   [0 0 1 0 1 1 0 1]  after one loop, get s1 is result of s0, now 2.elem
        s2's result:    [0 0 0 1 1 0 1 0]   [0 0 0 0 1 0 0 0]  now 4.elem
        s4's result:    [0 0 0 0 0 0 0 0]   [0 0 0 0 0 0 0 0]  end

        dst: 0.2.4. elem, so [1 0 1 0 1 0 0 0]
        so far, get output_mask_ub

    note:
        output mask: uint8
        output index: int32
        output proposals: float16 or float32
    """

    def __init__(self, tik_instance, all_inp_proposals_gm_1980, input_shape, input_dtype, iou_thres):
        """
        Parameters:
        ----------
        tik_instance: tik instance
        all_inp_proposals_gm_1980: size is N*8
        input_shape: corresponds to all_inp_proposals_ub_1980
        input_dtype: 1981 supports: float16 and float32
        iou_thres: iou threshold, one box is valid if its iou is lower than the threshold

        Returns
        -------
        None
        """
        self.tik_instance = tik_instance
        self.data_type = input_dtype

        if self.data_type == 'float16':
            self.mask = 256 // 2
            self.bytes_each_elem = 2
            self.vector_mask_max = 128
            self.RPN_PROPOSAL_NUM = 16
        elif self.data_type == 'float32':
            self.mask = 256 // 4
            self.bytes_each_elem = 4
            self.vector_mask_max = 64
            self.RPN_PROPOSAL_NUM = 8

        self.N, _ = input_shape
        # note: N canbe used in size, but not for def tensor, should use ceil_N
        self.input_size = self.N * ELEMENT_NUM

        # cache frequently used
        self.negone_int8_scalar = tik_instance.Scalar('int8', 'negone_int8_scalar', init_value=-1)
        self.zero_int8_scalar = tik_instance.Scalar('int8', 'zero_int8_scalar', init_value=0)
        self.zero_int16_scalar = tik_instance.Scalar('int16', 'zero_int16_scalar', init_value=0)
        self.one_int8_scalar = tik_instance.Scalar('int8', 'one_int8_scalar', init_value=1)
        self.one_int16_scalar = tik_instance.Scalar('int16', 'one_int16_scalar', init_value=1)

        # scalar: zero of dtype, one
        self.zero_datatype_scalar = tik_instance.Scalar(self.data_type, 'zero_dtype_scalar', init_value=0.)
        self.one_datatype_scalar = tik_instance.Scalar(self.data_type, 'one_dtype_scalar', init_value=1.)

        # note: defed size need to 32b aligned
        self.ceil_N = _ceiling(self.N, self.RPN_PROPOSAL_NUM)
        self.x1_ub = tik_instance.Tensor(shape=(self.ceil_N,), dtype=self.data_type, name='x1_ub', scope=tik.scope_ubuf)
        self.x2_ub = tik_instance.Tensor(shape=(self.ceil_N,), dtype=self.data_type, name='x2_ub', scope=tik.scope_ubuf)
        self.y1_ub = tik_instance.Tensor(shape=(self.ceil_N,), dtype=self.data_type, name='y1_ub', scope=tik.scope_ubuf)
        self.y2_ub = tik_instance.Tensor(shape=(self.ceil_N,), dtype=self.data_type, name='y2_ub', scope=tik.scope_ubuf)
        self.score_ub = tik_instance.Tensor(shape=(self.ceil_N,), dtype=self.data_type, name='score_ub',
                                            scope=tik.scope_ubuf)

        # 1980's input => 1981'soutput_mask_ub
        all_inp_proposals_ub_1980 = tik_instance.Tensor(self.data_type, (self.ceil_N, ELEMENT_NUM),
                                                        name="all_inp_proposals_ub_1980", scope=tik.scope_ubuf)
        # max. burst is 65535, so max. bytes is 65535*32b, support max. N is 65535*32/2/8=131070 for fp16
        tik_instance.data_move(all_inp_proposals_ub_1980, all_inp_proposals_gm_1980, 0, nburst=1,
                               burst=(self.ceil_N * ELEMENT_NUM * self.bytes_each_elem // 32),
                               src_stride=0, dst_stride=0)

        self._input_trans(all_inp_proposals_ub_1980)
        # cache area, calc once is enough
        self.total_areas_ub = None
        self.thres_ub = tik_instance.Tensor(shape=(self.ceil_N,), dtype=self.data_type, name="thres_ub",
                                            scope=tik.scope_ubuf)
        self._tailing_handle_vector_dup(self.thres_ub,
                                        scalar=tik_instance.Scalar(self.data_type, 'thres_scalar',
                                                                   init_value=iou_thres),
                                        size=self.ceil_N)

        # [0] stores next nonzero idx, 32 for 32b aligned
        self.next_nonzero_int8_idx = tik_instance.Tensor('int8', (32,), tik.scope_ubuf,
                                                         'next_nonzero_int8_idx')

        # for update_valid_mask, valid_mask uses int16, which is for using vand, but 920 doesnot support fp162int16
        # as using int8, so here 32/1=32
        self.valid_mask_size_int8 = _ceiling(self.N, 32)
        self.valid_mask_int8_ub = tik_instance.Tensor('int8', (self.valid_mask_size_int8,), tik.scope_ubuf,
                                                      'valid_mask_int8_ub')
        with tik_instance.for_range(0, self.valid_mask_size_int8) as i:
            # init with all 1, which means all is valid at the beginning
            self.valid_mask_int8_ub[i] = self.one_int8_scalar

        # update valid mask, here float16 fixed, ensure 32b aligned. note: size below = valid_mask_size_int8
        self.tmp_valid_mask_float16 = self.tik_instance.Tensor('float16', (self.valid_mask_size_int8,), tik.scope_ubuf,
                                                               'tmp_valid_mask_float16')
        self.tmp_mask_float16 = self.tik_instance.Tensor('float16', (self.valid_mask_size_int8,), tik.scope_ubuf,
                                                         'tmp_mask')

        # selected_boxes and idx generate
        self.selected_boxes_ub = self._selected_boxes_gen(all_inp_proposals_ub_1980)
        self.selected_idx_ub = self._selected_idx_gen()

        # for iou
        self.out_iou_ub = self.tik_instance.Tensor(shape=(self.ceil_N,), dtype=self.data_type, name='out_iou_ub',
                                                   scope=tik.scope_ubuf)

        # for inter
        self._init_for_inter()

        # for union
        self.out_union = self.tik_instance.Tensor(shape=(self.ceil_N,), dtype=self.data_type, name='union_ub',
                                                  scope=tik.scope_ubuf)

        # output mask, dtype is int8 fixed
        self.output_mask_int8_ub = self.tik_instance.Tensor('int8', (self.valid_mask_size_int8,), tik.scope_ubuf,
                                                            "output_mask_int_ub")

        self._init_for_cmpmask2bitmask()

        # scaling
        self._scaling()

    def _init_for_inter(self):
        """
        init tensors for inter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.xx1 = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "xx1_ub")
        self.yy1 = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "yy1_ub")
        self.xx2 = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "xx2_ub")
        self.yy2 = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "yy2_ub")
        self.x1i = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "x1i_ub")
        self.y1i = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "y1i_ub")
        self.x2i = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "x2i_ub")
        self.y2i = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "y2i_ub")
        self.w = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, 'w_ub')
        self.h = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, 'h_ub')
        self.zeros = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, 'zeros_ub')
        self._tailing_handle_vector_dup(self.zeros,
                                        scalar=self.tik_instance.Scalar(self.data_type, 'tmp_s1', init_value=0.),
                                        size=self.ceil_N)
        self.inter = self.tik_instance.Tensor(self.data_type, (self.ceil_N,), tik.scope_ubuf, "inter_ub")

    def _init_for_cmpmask2bitmask(self):
        """
        for cmpmask2bitmask, fp16 fixed is OK, this is used in one repeat, so 128 below is OK

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.output_mask_f16 = self.tik_instance.Tensor('float16', (128,), name="output_mask_f16",
                                                        scope=tik.scope_ubuf)
        zero_fp16_scalar = self.tik_instance.Scalar(dtype="float16", name="zero_scalar", init_value=0.0)
        one_fp16_scalar = self.tik_instance.Scalar(dtype="float16", name="one_scalar", init_value=1.0)
        self.data_fp16_zero = self.tik_instance.Tensor("float16", (128,), name="data_zero", scope=tik.scope_ubuf)
        self.data_fp16_one = self.tik_instance.Tensor("float16", (128,), name="data_one", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(128, self.data_fp16_zero, zero_fp16_scalar, 1, 1, 8)
        self.tik_instance.vector_dup(128, self.data_fp16_one, one_fp16_scalar, 1, 1, 8)

    def _input_trans(self, all_inp_proposals_ub_1980):
        """
        1980's inputs trans to 1981's
        Note: should use vreduce, not vgather

        Parameters
        ----------
        all_inp_proposals_ub_1980:
            1980:
                shape is (N, 8), only one addr_base
                [
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                ...
                ]

            1981:
                5 addr_bases
                x1[] with N elems
                x2[]
                y1[]
                y2[]
                score[]

        Returns
        -------
        None
        """
        if self.data_type == 'float16':

            pattern_x1 = self.tik_instance.Tensor('uint16', (16,), tik.scope_ubuf,
                                                  name='pattern_x1_ub')
            pattern_y1 = self.tik_instance.Tensor('uint16', (16,), tik.scope_ubuf,
                                                  name='pattern_y1_ub')
            pattern_x2 = self.tik_instance.Tensor('uint16', (16,), tik.scope_ubuf,
                                                  name='pattern_x2_ub')
            pattern_y2 = self.tik_instance.Tensor('uint16', (16,), tik.scope_ubuf,
                                                  name='pattern_y2_ub')

            self.tik_instance.vector_dup(16, pattern_x1,
                                         self.tik_instance.Scalar('uint16', init_value=PATTERN_VALUE_FP16_X1), 1, 1, 1)
            self.tik_instance.vector_dup(16, pattern_y1,
                                         self.tik_instance.Scalar('uint16', init_value=PATTERN_VALUE_FP16_Y1), 1, 1, 1)
            self.tik_instance.vector_dup(16, pattern_x2,
                                         self.tik_instance.Scalar('uint16', init_value=PATTERN_VALUE_FP16_X2), 1, 1, 1)
            self.tik_instance.vector_dup(16, pattern_y2,
                                         self.tik_instance.Scalar('uint16', init_value=PATTERN_VALUE_FP16_Y2), 1, 1, 1)
        else:
            # fp32
            pattern_x1 = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf, name='pattern_x1_ub')
            pattern_y1 = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf, name='pattern_y1_ub')
            pattern_x2 = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf, name='pattern_x2_ub')
            pattern_y2 = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf, name='pattern_y2_ub')

            self.tik_instance.vector_dup(8, pattern_x1, self.tik_instance.Scalar('uint32',
                                                                                 init_value=PATTERN_VALUE_FP32_X1),
                                         1, 1, 1)
            self.tik_instance.vector_dup(8, pattern_y1, self.tik_instance.Scalar('uint32',
                                                                                 init_value=PATTERN_VALUE_FP32_Y1),
                                         1, 1, 1)
            self.tik_instance.vector_dup(8, pattern_x2, self.tik_instance.Scalar('uint32',
                                                                                 init_value=PATTERN_VALUE_FP32_X2),
                                         1, 1, 1)
            self.tik_instance.vector_dup(8, pattern_y2, self.tik_instance.Scalar('uint32',
                                                                                 init_value=PATTERN_VALUE_FP32_Y2),
                                         1, 1, 1)

        self._tailing_handle_vreduce_input(self.x1_ub, all_inp_proposals_ub_1980, pattern_x1)
        self._tailing_handle_vreduce_input(self.y1_ub, all_inp_proposals_ub_1980, pattern_y1)
        self._tailing_handle_vreduce_input(self.x2_ub, all_inp_proposals_ub_1980, pattern_x2)
        self._tailing_handle_vreduce_input(self.y2_ub, all_inp_proposals_ub_1980, pattern_y2)

    def _tailing_handle_vreduce_input(self, dst_ub, src0_ub, src1_pattern_ub):
        """
        tailing handle: means handle all inputs, especially the tail is special and need to deal with
        3 steps to handle tailing.
        for nms: step2 and step3 is enough, if all uses UB at the same time

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0 in ub
        src1_pattern_ub: pattern for src1

        Returns
        -------
        None
        """
        # 16 for fp16, 8 for fp32
        vector_proposals_max = self.vector_mask_max // 8
        offset = 0

        # step2: repeat?
        repeat = self.ceil_N % (vector_proposals_max * REPEAT_TIMES_MAX) // vector_proposals_max
        if repeat > 0:
            self.tik_instance.vreduce(mask=self.vector_mask_max,
                                      dst=dst_ub[offset],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=repeat,
                                      src0_blk_stride=1,
                                      src0_rep_stride=self.vector_mask_max * self.bytes_each_elem // 32,
                                      # here 0 means: pattern is reused in each repeat
                                      src1_rep_stride=0)

        # step3: last num?
        last_num = self.ceil_N % vector_proposals_max
        if last_num > 0:
            offset += repeat * vector_proposals_max
            self.tik_instance.vreduce(mask=8 * last_num,
                                      dst=dst_ub[offset],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=1,
                                      src0_blk_stride=1,
                                      # no need to repeat, so 0
                                      src0_rep_stride=0,
                                      # here 0 means: pattern is reused in each repeat
                                      src1_rep_stride=0)

    def _tailing_handle_vreduce_output(self, dst_ub, src0_ub, src1_pattern_ub):
        """
        [N, 8] => [N, 5]

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0 in ub
        src1_pattern_ub: pattern for src1

        Returns
        -------
        None
        """
        # =16 for fp16, =8 for fp32. here 8 is ncols
        vector_proposals_max = self.vector_mask_max // 8
        offset = 0

        # step2: repeat?
        repeat = self.ceil_N % (vector_proposals_max * REPEAT_TIMES_MAX) // vector_proposals_max
        if repeat > 0:
            self.tik_instance.vreduce(mask=self.vector_mask_max,
                                      dst=dst_ub[offset * 5],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=repeat,
                                      src0_blk_stride=1,
                                      src0_rep_stride=self.vector_mask_max * self.bytes_each_elem // 32,
                                      src1_rep_stride=0)

        # step3: last num?
        last_num = self.ceil_N % vector_proposals_max
        if last_num > 0:
            offset += repeat * vector_proposals_max
            self.tik_instance.vreduce(mask=8 * last_num,
                                      dst=dst_ub[offset * 5],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=1,
                                      src0_blk_stride=1, src0_rep_stride=0,
                                      src1_rep_stride=0)

    def _tailing_handle_vmuls(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vmuls

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src ub
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat?
        repeat = size % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vmuls(mask=self.vector_mask_max,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=repeat,
                                    dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=8, src_rep_stride=8)

        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vmuls(mask=last_num,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=8, src_rep_stride=8)

    def _tailing_handle_vector_dup(self, dst_ub, scalar, size):
        """
        handle tailing of vector_dup

        Parameters
        ----------
        dst_ub: dst tensor in ub
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat?
        repeat = size % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vector_dup(mask=self.vector_mask_max,
                                         dst=dst_ub[offset],
                                         scalar=scalar,
                                         repeat_times=repeat,
                                         dst_blk_stride=1, dst_rep_stride=8)

        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vector_dup(mask=last_num,
                                         dst=dst_ub[offset],
                                         scalar=scalar,
                                         repeat_times=1,
                                         dst_blk_stride=1, dst_rep_stride=8)

    def _tailing_handle_vmax(self, dst_ub, src0_ub, src1_ub, size):
        """
        handle tailing of vmax

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat?
        repeat = size % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vmax(mask=self.vector_mask_max,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=repeat,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8,
                                   # 8 is fixed here for both fp16 and fp32
                                   src1_rep_stride=8)

        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vmax(mask=last_num,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=1,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)

    def _tailing_handle_vmin(self, dst_ub, src0_ub, src1_ub, size):
        """
        handle tailing of vmin

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat?
        repeat = size % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vmin(mask=self.vector_mask_max,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=repeat,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8,
                                   src1_rep_stride=8)

        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vmin(mask=last_num,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=1,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)

    def _tailing_handle_vsub(self, dst_ub, src0_ub, src1_ub, size):
        """
        handle tailing of vsub

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat?
        repeat = size % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vsub(mask=self.vector_mask_max,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=repeat,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8,
                                   src1_rep_stride=8)

        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vsub(mask=last_num,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=1,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)

    def _tailing_handle_vmul(self, dst_ub, src0_ub, src1_ub, size, mask_max=None):
        """
        handle tailing of vmul

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems
        mask_max: max. mask

        Returns
        -------
        None
        """
        if mask_max is None:
            mask_max = self.vector_mask_max

        offset = 0

        # step2: repeat?
        repeat = size % (mask_max * REPEAT_TIMES_MAX) // mask_max
        if repeat > 0:
            self.tik_instance.vmul(mask=mask_max,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=repeat,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8,
                                   src1_rep_stride=8)

        # step3: last num?
        last_num = size % mask_max
        if last_num > 0:
            offset += repeat * mask_max
            self.tik_instance.vmul(mask=last_num,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=1,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)

    def _tailing_handle_vadds(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vadds

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat?
        repeat = size % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vadds(mask=self.vector_mask_max,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=repeat,
                                    dst_blk_stride=1, src_blk_stride=1,
                                    dst_rep_stride=8, src_rep_stride=8)

        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vadds(mask=last_num,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1, src_blk_stride=1,
                                    dst_rep_stride=8, src_rep_stride=8)

    def _tailing_handle_vrec(self, dst_ub, src_ub, size):
        """
        handle tailing of vrec

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat?
        repeat = size % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vrec(mask=self.vector_mask_max,
                                   dst=dst_ub[offset],
                                   src=src_ub[offset],
                                   repeat_times=repeat,
                                   dst_blk_stride=1, src_blk_stride=1,
                                   dst_rep_stride=8, src_rep_stride=8)

        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vrec(mask=last_num,
                                   dst=dst_ub[offset],
                                   src=src_ub[offset],
                                   repeat_times=1,
                                   dst_blk_stride=1, src_blk_stride=1,
                                   dst_rep_stride=8, src_rep_stride=8)

    def _tailing_handle_vdiv(self, dst_ub, src0_ub, src1_ub, size):
        """
        handle tailing of vdiv

        Parameters
        ----------
        dst_ub:
        src0_ub:
        src1_ub:
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat?
        repeat = size % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vdiv(mask=self.vector_mask_max,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=repeat,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8,
                                   src1_rep_stride=8)

        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vdiv(mask=last_num,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=1,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)

    def _tailing_handle_vec_conv(self, dst_ub, src_ub, size, dst_bytes, src_bytes):
        """
        handle tailing of vec_conv

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src ub
        size: totol size of elems
        dst_bytes: bytes of each elem of dst
        src_bytes: bytes of each elem of src

        Returns
        -------
        None
        """
        # max. is 128. src_bytes can be 1
        mask_max = min(256 // src_bytes, 128)
        offset = 0

        # step2: repeat?
        repeat = size % (mask_max * REPEAT_TIMES_MAX) // mask_max
        if repeat > 0:
            self.tik_instance.vec_conv(mask=mask_max,
                                       mode="none",
                                       dst=dst_ub[offset],
                                       src=src_ub[offset],
                                       repeat_times=repeat,
                                       dst_rep_stride=mask_max * dst_bytes // 32,
                                       src_rep_stride=mask_max * src_bytes // 32)

        # step3: last num?
        last_num = size % mask_max
        if last_num > 0:
            offset += repeat * mask_max
            self.tik_instance.vec_conv(mask=last_num,
                                       mode="none",
                                       dst=dst_ub[offset],
                                       src=src_ub[offset],
                                       repeat_times=1,
                                       dst_rep_stride=0, src_rep_stride=0)

    def _selected_boxes_gen(self, proposals_ub_1980):
        """
        selected_boxes generate from proposals_ub_1980

        original box_scores: [N, 8]
        selected_boxes:      [N, 5]

        Parameters
        ----------
        proposals_ub_1980: input proposals

        Returns
        -------
        selected_boxes_ub:
        """
        # def selected_boxes_ub
        selected_boxes_ub = self.tik_instance.Tensor(self.data_type, (self.ceil_N, 5), tik.scope_ubuf,
                                                     'selected_boxes_ub')

        # do
        if self.data_type == 'float16':
            pattern = self.tik_instance.Tensor('uint16', (16,), tik.scope_ubuf,
                                               'pattern_ub')
            # remember: init pattern
            self.tik_instance.vector_dup(16, pattern,
                                         self.tik_instance.Scalar('uint16', 'pattern_s', init_value=PATTERN_VALUE_7967),
                                         1, 1, 1)
        else:
            pattern = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf,
                                               'pattern_ub')
            self.tik_instance.vector_dup(8, pattern,
                                         self.tik_instance.Scalar('uint32', 'pattern_s',
                                                                  init_value=PATTERN_VALUE_522133279), 1, 1, 1)

        self._tailing_handle_vreduce_output(selected_boxes_ub, proposals_ub_1980, pattern)

        return selected_boxes_ub

    def _selected_idx_gen(self):
        """
        selected_idx generate

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # int32 is fixed for output index
        selected_idx_ub = self.tik_instance.Tensor('int32', (self.ceil_N,), tik.scope_ubuf,
                                                   'selected_idx_ub')

        # consider: optimize, gm2ub. def and init tensor in gm
        idx_list = list(range(self.ceil_N))
        selected_idx_gm = self.tik_instance.Tensor('int32', (self.ceil_N,), tik.scope_gm, 'selected_idx_gm',
                                                   init_value=idx_list)
        self.tik_instance.data_move(selected_idx_ub, selected_idx_gm, 0, nburst=1,
                                    burst=self.ceil_N * 4 // 32,
                                    src_stride=0, dst_stride=0)

        return selected_idx_ub

    def _scaling(self):
        """
        scaling of input, scaling factor is DOWN_FACTOR

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._tailing_handle_vmuls(self.x1_ub, self.x1_ub, DOWN_FACTOR, self.ceil_N)
        self._tailing_handle_vmuls(self.x2_ub, self.x2_ub, DOWN_FACTOR, self.ceil_N)
        self._tailing_handle_vmuls(self.y1_ub, self.y1_ub, DOWN_FACTOR, self.ceil_N)
        self._tailing_handle_vmuls(self.y2_ub, self.y2_ub, DOWN_FACTOR, self.ceil_N)

    def _area(self):
        """
        area = (x2-x1) * (y2-y1), this is vector computing
        area can be reused in loops

        Parameters
        ----------
        None


        Returns
        -------
        None
        """
        if not self.total_areas_ub is None:
            return self.total_areas_ub

        tik_instance = self.tik_instance
        data_type = self.x1_ub.dtype
        self.total_areas_ub = tik_instance.Tensor(data_type, (self.ceil_N,), name="total_areas_ub",
                                                  scope=tik.scope_ubuf)

        x2subx1 = tik_instance.Tensor(shape=(self.ceil_N,), dtype=data_type, name='x2subx1', scope=tik.scope_ubuf)
        y2suby1 = tik_instance.Tensor(shape=(self.ceil_N,), dtype=data_type, name='y2suby1', scope=tik.scope_ubuf)

        self._tailing_handle_vsub(x2subx1, self.x2_ub, self.x1_ub, self.ceil_N)
        self._tailing_handle_vsub(y2suby1, self.y2_ub, self.y1_ub, self.ceil_N)
        self._tailing_handle_vmul(self.total_areas_ub, x2subx1, y2suby1, self.ceil_N)

        return self.total_areas_ub

    def _intersection(self, cur):
        """
        intersection calculation

        Parameters
        ----------
        cur: intersection of cur proposal and the others

        Returns
        -------
        None
        """
        self._tailing_handle_vector_dup(self.x1i,
                                        self.tik_instance.Scalar(self.data_type, 'x1_s', init_value=self.x1_ub[cur]),
                                        self.ceil_N)
        self._tailing_handle_vector_dup(self.y1i,
                                        self.tik_instance.Scalar(self.data_type, 'y1_s', init_value=self.y1_ub[cur]),
                                        self.ceil_N)
        self._tailing_handle_vector_dup(self.x2i,
                                        self.tik_instance.Scalar(self.data_type, 'x2_s', init_value=self.x2_ub[cur]),
                                        self.ceil_N)
        self._tailing_handle_vector_dup(self.y2i,
                                        self.tik_instance.Scalar(self.data_type, 'y2_s', init_value=self.y2_ub[cur]),
                                        self.ceil_N)

        # xx1 = max(x1[i], x1[1:]),  yy1 = max(y1[i], y1[1:]), xx2=min(x2[i], x2[1:]),  yy2=min(y2[i], y2[1:])
        self._tailing_handle_vmax(self.xx1, self.x1_ub, self.x1i, self.ceil_N)
        self._tailing_handle_vmax(self.yy1, self.y1_ub, self.y1i, self.ceil_N)
        self._tailing_handle_vmin(self.xx2, self.x2_ub, self.x2i, self.ceil_N)
        self._tailing_handle_vmin(self.yy2, self.y2_ub, self.y2i, self.ceil_N)

        # w = max(0, xx2-xx1+offset), h = max(0, yy2-yy1+offset), offset=0 here
        self._tailing_handle_vsub(self.xx1, self.xx2, self.xx1, self.ceil_N)
        self._tailing_handle_vmax(self.w, self.xx1, self.zeros, self.ceil_N)
        self._tailing_handle_vsub(self.yy1, self.yy2, self.yy1, self.ceil_N)
        self._tailing_handle_vmax(self.h, self.yy1, self.zeros, self.ceil_N)
        self._tailing_handle_vmul(self.inter, self.w, self.h, self.ceil_N)

        return self.inter

    def _union(self, cur):
        """
        union(A, B) = A + B - inter(A, B)
        for K[0] and K[:]

        Parameters
        ----------
        cur: intersection of cur proposal and the others

        Returns
        -------
        None
        """
        areas = self._area()
        inter = self._intersection(cur)

        area_i = self.tik_instance.Scalar(self.data_type, 'k0_scalar', init_value=areas[cur])
        self._tailing_handle_vadds(self.out_union, areas, area_i, self.ceil_N)
        self._tailing_handle_vsub(self.out_union, self.out_union, inter, self.ceil_N)

        return self.out_union

    def _iou(self, cur):
        """
        intersection of union

        precision of vrec is a problem, diff >= 0.001
        iou = inter / union

        Parameters
        ----------
        cur: compute cur proposal and the others

        Returns
        -------
        None
        """
        tmp_inter = self._intersection(cur)
        tmp_union = self._union(cur)

        self._tailing_handle_vrec(tmp_union, tmp_union, self.ceil_N)
        self._tailing_handle_vmul(self.out_iou_ub, tmp_inter, tmp_union, self.ceil_N)

        return self.out_iou_ub

    def _cmpmask2bitmask(self, dst_ub, cmpmask, handle_dst_size):
        """
        in one repeat, handle max. 128 elems. so tensor defed below has 128 shape
        bitmask is like [1 0 1 1 0 0 0 1]

        Parameters
        ----------
        cur: compute cur proposal and the others

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        tik_instance.vsel(128, 0, self.output_mask_f16, cmpmask, self.data_fp16_one, self.data_fp16_zero, 1, 1, 1, 1, 8,
                          8, 8)

        tik_instance.vec_conv(handle_dst_size, "none", dst_ub, self.output_mask_f16, 1, 8, 8)

    def _update_next_nonzero_idx(self, mask, begin_idx):
        """
        find next nonzero idx
        note: use scalar instead of tensor(next_nonzero_int8_idx)

        Parameters
        ----------
        mask: is [0 0 1 1 ], its size is ceilN
        begin_idx: from where begin loop

        Returns
        -------
        tensor[0] with only one elem.
        """
        self.next_nonzero_int8_idx[0].set_as(self.negone_int8_scalar)
        # tensor set_as is slow
        with self.tik_instance.for_range(begin_idx, self.N) as i:
            with self.tik_instance.if_scope(mask[i] == 1):
                with self.tik_instance.if_scope(self.next_nonzero_int8_idx[0] == -1):
                    self.next_nonzero_int8_idx[0] = i

    def _one_loop(self, cur):
        """
        in one loop: iou, generate bitmask and return output_mask_int8_ub

        Parameters
        ----------
        cur: compute cur proposal and the others

        Returns
        -------
        output_mask_int8_ub
        """
        out_iou = self._iou(cur)

        # cmpmask 2 bitmask
        output_mask_int8_ub = self._tailing_handle_cmp_le_and_2bitmask(out_iou, self.thres_ub,
                                                                       self.ceil_N)

        # set output_mask[cur] = 0, because will be added into DST, and deleted from SRC proposal list
        output_mask_int8_ub[cur].set_as(self.zero_int8_scalar)
        return output_mask_int8_ub

    def _tailing_handle_cmp_le_and_2bitmask(self, src0_ub, src1_ub, size):
        """
        combine vcmp_le() and cmpmask2bitmask()
        vcmp handle max. 128 mask, repeat = 1

        size: total size of proposals

        Parameters
        ----------
        src0_ub: src0 in ub
        src1_ub: src1 in ub
        cur: compute cur proposal and the others

        Returns
        -------
        output_mask_int8_ub
        """
        loops = size // (self.vector_mask_max * 1)
        offset = 0

        # step1: max. mask * max. repeat  * loops times
        if loops > 0:
            for loop_index in range(0, loops):
                # vcmp only run once, so repeat = 1
                cmpmask = self.tik_instance.vcmp_le(mask=self.vector_mask_max,
                                                    src0=src0_ub[offset],
                                                    src1=src1_ub[offset],
                                                    # 1 is fixed
                                                    src0_stride=1, src1_stride=1)
                self._cmpmask2bitmask(dst_ub=self.output_mask_int8_ub[offset],
                                      cmpmask=cmpmask, handle_dst_size=self.vector_mask_max)

                offset = (loop_index + 1) * self.vector_mask_max * 1

        # step2: not used
        # step3: last num?
        last_num = size % self.vector_mask_max
        if last_num > 0:
            cmpmask = self.tik_instance.vcmp_le(mask=last_num,
                                                src0=src0_ub[offset],
                                                src1=src1_ub[offset],
                                                src0_stride=1, src1_stride=1)
            self._cmpmask2bitmask(dst_ub=self.output_mask_int8_ub[offset],
                                  cmpmask=cmpmask, handle_dst_size=last_num)

        return self.output_mask_int8_ub

    def _update_valid_mask(self, mask_ub_int8_ub):
        """
        update valid mask
        note: use vand instead of vmul, but vand only compute uint16/int16,
            so use int16 for out_mask in 1981. but 1981 donot support f162s16 in cmpmask2bitmask()

        Parameters
        ----------
        mask_ub_int8_ub: which will be used to update valid_mask_ub

        Returns
        -------
        None
        """
        self._tailing_handle_vec_conv(self.tmp_valid_mask_float16, self.valid_mask_int8_ub, self.valid_mask_size_int8,
                                      dst_bytes=2, src_bytes=1)
        self._tailing_handle_vec_conv(self.tmp_mask_float16, mask_ub_int8_ub, self.valid_mask_size_int8, dst_bytes=2,
                                      src_bytes=1)

        # [0 0 1 1] * [1 0 1 0] = [0 0 1 0]
        self._tailing_handle_vmul(self.tmp_valid_mask_float16, self.tmp_valid_mask_float16, self.tmp_mask_float16,
                                  self.valid_mask_size_int8, mask_max=128)

        # float16 to int8
        self._tailing_handle_vec_conv(self.valid_mask_int8_ub, self.tmp_valid_mask_float16, self.valid_mask_size_int8,
                                      dst_bytes=1, src_bytes=2)

    def loops(self):
        """
        run loops

        Parameters
        ----------
        None

        Returns
        -------
        selected_mask_ub
        """
        # def and init selected_mask_ub
        selected_mask_ub = self.tik_instance.Tensor('int8', (self.ceil_N,), name="selected_mask_ub",
                                                    scope=tik.scope_ubuf)

        # avoid scalar op on device, use host op instead as below
        selected_mask_gm_list = [0] * self.ceil_N
        selected_mask_gm = self.tik_instance.Tensor('int8', (self.ceil_N,), name="selected_mask_gm", scope=tik.scope_gm,
                                                    init_value=selected_mask_gm_list)
        self.tik_instance.data_move(selected_mask_ub, selected_mask_gm, 0, nburst=1,
                                    burst=_ceiling(self.ceil_N * 1, 32) // 32,
                                    # ceil_n canbe 16, so needs ceiling here
                                    src_stride=0, dst_stride=0)

        # init state
        selected_mask_ub[self.zero_int8_scalar] = self.one_int8_scalar
        output_mask_int8_ub = self._one_loop(self.zero_int8_scalar)
        self._update_valid_mask(output_mask_int8_ub)
        self._update_next_nonzero_idx(self.valid_mask_int8_ub,
                                      begin_idx=self.zero_int8_scalar)
        start_loop_idx = self.tik_instance.Scalar(dtype='int8', name='dst_idx',
                                                  init_value=self.next_nonzero_int8_idx[0])

        # plan: consider: avoid scalar op
        cur = self.tik_instance.Scalar(dtype='int8', name='cur_scalar')
        # use all loops to ensure
        with self.tik_instance.for_range(start_loop_idx, self.N):
            cur.set_as(self.next_nonzero_int8_idx[0])

            with self.tik_instance.if_scope(0 <= cur < self.N):
                # set 1, means valid
                selected_mask_ub[cur] = self.one_int8_scalar
                mask_ub = self._one_loop(cur)
                self._update_valid_mask(mask_ub)
                self._update_next_nonzero_idx(self.valid_mask_int8_ub, begin_idx=cur)

        return selected_mask_ub


# pylint: disable=too-many-locals,too-many-arguments
def _tik_func_nms_single_core_multithread_1981(input_shape, input_dtype, thresh, total_output_proposal_num,
                                               kernel_name_var):
    """
    Compute output boxes after non-maximum suppression for 1981

    Parameters
    ----------
    input_shape: dict
        shape of input boxes, including proposal boxes and corresponding confidence scores

    input_dtype: str
        input data type: options are float16 and float32

    thresh: float
        iou threshold

    total_output_proposal_num: int
        the number of output proposal boxes

    kernel_name: str
        cce kernel name

    Returns
    -------
    tik_instance: TIK API
    """
    tik_instance = tik.Tik()
    total_input_proposal_num, _ = input_shape
    proposals = tik_instance.Tensor(input_dtype, (total_input_proposal_num, ELEMENT_NUM),
                                    name="in_proposals",
                                    scope=tik.scope_gm)

    nms_helper = _NMSHelper(tik_instance, proposals, (total_input_proposal_num, ELEMENT_NUM), input_dtype,
                            thresh)
    output_proposals_ub = nms_helper.selected_boxes_ub
    output_index_ub = nms_helper.selected_idx_ub
    output_mask_ub = nms_helper.loops()

    # data move from ub to gm. def tensor in gm can be real shape, dont need to ceiling
    out_proposals_gm = tik_instance.Tensor(input_dtype, (total_output_proposal_num, VALID_COLUMN_NUM),
                                           name="out_proposals_gm", scope=tik.scope_gm)
    # address is 32B aligned
    out_index_gm = tik_instance.Tensor("int32", (total_output_proposal_num,), name="out_index_gm", scope=tik.scope_gm)
    out_mask_gm = tik_instance.Tensor("uint8", (total_output_proposal_num,), name="out_mask_gm", scope=tik.scope_gm)

    tik_instance.data_move(out_proposals_gm, output_proposals_ub, 0, nburst=1,
                           burst=(nms_helper.ceil_N * VALID_COLUMN_NUM * nms_helper.bytes_each_elem // 32),
                           # max. burst is 65535, unit is 32B, so support: 65535*32/2/8=131070 proposals if fp16.
                           src_stride=0, dst_stride=0)
    tik_instance.data_move(out_index_gm, output_index_ub, 0, nburst=1,
                           burst=(nms_helper.ceil_N * 4 // 32),
                           src_stride=0, dst_stride=0)
    tik_instance.data_move(out_mask_gm, output_mask_ub, 0, nburst=1,
                           burst=_ceiling(nms_helper.ceil_N * 1, 32) // 32,
                           # here need _ceiling() as ceilN can be 16; 16*1//32=0 is wrong
                           src_stride=0, dst_stride=0)

    tik_instance.BuildCCE(kernel_name=kernel_name_var,
                          inputs=[proposals],
                          outputs=[out_proposals_gm, out_index_gm, out_mask_gm],
                          output_files_path=None,
                          enable_l2=False)
    return tik_instance


# pylint: disable=unused-argument,too-many-locals,too-many-arguments
def _nms_with_mask_1981(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr, kernel_name="nms_with_mask"):
    """
    algorithm: nms_with_mask 1981

    find the best target bounding box and eliminate redundant bounding boxes

    Parameters
    ----------
    box_scores: dict
        2-D shape and dtype of input tensor, only support [N, 8]
        including proposal boxes and corresponding confidence scores

    selected_boxes: dict
        2-D shape and dtype of output boxes tensor, only support [N,5]
        including proposal boxes and corresponding confidence scores

    selected_idx: dict
        the index of input proposal boxes

    selected_mask: dict
        the symbol judging whether the output proposal boxes is valid

    iou_thr: float
        iou threshold

    kernel_name: str
        cce kernel name, default value is "nms_with_mask"

    Returns
    -------
    None
    """
    input_shape = box_scores.get("shape")
    input_dtype = box_scores.get("dtype").lower()

    # check dtype
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="box_scores")
    # check shape
    para_check.check_shape(input_shape, min_rank=INPUT_DIM, max_rank=INPUT_DIM, param_name="box_scores")

    output_size, _ = input_shape
    return _tik_func_nms_single_core_multithread_1981(input_shape, input_dtype, iou_thr, output_size, kernel_name)
