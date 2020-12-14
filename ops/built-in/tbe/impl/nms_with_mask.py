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
    src0_ub = ib.Tensor("float16", (BURST_PROPOSAL_NUM, ), name="src0_ub", scope=tik.scope_ubuf)
    src1_ub = ib.Tensor("float16", (BURST_PROPOSAL_NUM, ), name="src1_ub", scope=tik.scope_ubuf)
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

    kernel_name: str
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
    out_index = tik_instance.Tensor("int32", (_ceil_div(total_output_proposal_num, ELEMENT_NUM) * ELEMENT_NUM, ),
                                    name="out_index",
                                    scope=tik.scope_gm)
    out_mask = tik_instance.Tensor("uint8",
                                   (_ceil_div(total_output_proposal_num, CONFIG_DATA_ALIGN) * CONFIG_DATA_ALIGN, ),
                                   name="out_mask",
                                   scope=tik.scope_gm)
    output_index_ub = tik_instance.Tensor("int32", (BURST_PROPOSAL_NUM, ), name="output_index_ub", scope=tik.scope_ubuf)
    output_mask_ub = tik_instance.Tensor("uint8", (BURST_PROPOSAL_NUM, ), name="output_mask_ub", scope=tik.scope_ubuf)
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
    selected_area_ub = tik_instance.Tensor("float16", (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM), ),
                                           name="selected_area_ub",
                                           scope=tik.scope_ubuf)
    # init middle sup_vec
    sup_vec_ub = tik_instance.Tensor("uint16", (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM), ),
                                     name="sup_vec_ub",
                                     scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, sup_vec_ub[0], 1, 1, 1, 8)

    # init nms tensor
    temp_area_ub = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM, ), name="temp_area_ub", scope=tik.scope_ubuf)
    temp_iou_ub = tik_instance.Tensor("float16",
                                      (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM), RPN_PROPOSAL_NUM),
                                      name="temp_iou_ub",
                                      scope=tik.scope_ubuf)
    temp_join_ub = tik_instance.Tensor("float16",
                                       (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM), RPN_PROPOSAL_NUM),
                                       name="temp_join_ub",
                                       scope=tik.scope_ubuf)
    temp_sup_matrix_ub = tik_instance.Tensor("uint16", (_ceiling(total_output_proposal_num, RPN_PROPOSAL_NUM), ),
                                             name="temp_sup_matrix_ub",
                                             scope=tik.scope_ubuf)
    temp_sup_vec_ub = tik_instance.Tensor("uint16", (BURST_PROPOSAL_NUM, ),
                                          name="temp_sup_vec_ub",
                                          scope=tik.scope_ubuf)

    if support_vreduce and support_v4dtrans:
        output_mask_f16 = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM, ),
                                              name="output_mask_f16",
                                              scope=tik.scope_ubuf)
        data_zero, data_one = _get_src_tensor(tik_instance)

        middle_reduced_proposals = tik_instance.Tensor("float16", (BURST_PROPOSAL_NUM, ELEMENT_NUM),
                                                       name="middle_reduced_proposals",
                                                       scope=tik.scope_ubuf)

        # init v200 reduce param
        nms_tensor_pattern = tik_instance.Tensor(dtype="uint16",
                                                 shape=(ELEMENT_NUM, ),
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
    input_shape = box_scores.get("shape")
    input_dtype = box_scores.get("dtype").lower()

    # check dtype
    check_list = ("float16")
    para_check.check_dtype(input_dtype, check_list, param_name="box_scores")
    #check shape
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
    _tik_func_nms_single_core_multithread(input_shape, iou_thr, output_size, kernel_name)
