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


def _ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


# pylint: disable=invalid-name
def _get_reduced_proposal(ib, out_proposal, in_proposal):
    """
    Reduce input proposal when input boxes out of range.

    Parameters
    ----------
    ib: TIK API

    out_proposal: output proposal after reduce

    in_proposal: input proposal with boxes and scores

    Returns
    -------
    out_proposal
    """
    coord_addr = ib.Tensor("float16", [4, BURST_PROPOSAL_NUM], name="coord_addr", scope=tik.scope_ubuf)

    ib.vextract(coord_addr[0], in_proposal[0], coord_addr.shape[1] // RPN_PROPOSAL_NUM, 0)  # x1
    ib.vextract(coord_addr[coord_addr.shape[1] * 1], in_proposal[0], coord_addr.shape[1] // RPN_PROPOSAL_NUM, 1)  # y1
    ib.vextract(coord_addr[coord_addr.shape[1] * 2], in_proposal[0], coord_addr.shape[1] // RPN_PROPOSAL_NUM, 2)  # x2
    ib.vextract(coord_addr[coord_addr.shape[1] * 3], in_proposal[0], coord_addr.shape[1] // RPN_PROPOSAL_NUM, 3)  # y2

    ib.vmuls(128, coord_addr[0], coord_addr[0], DOWN_FACTOR, coord_addr.shape[1] // BURST_PROPOSAL_NUM, 1, 1, 8, 8)
    # x1*DOWN_FACTOR
    ib.vmuls(128, coord_addr[coord_addr.shape[1] * 1], coord_addr[coord_addr.shape[1] * 1], DOWN_FACTOR,
             coord_addr.shape[1] // BURST_PROPOSAL_NUM, 1, 1, 8, 8)  # y1*DOWN_FACTOR
    ib.vmuls(128, coord_addr[coord_addr.shape[1] * 2], coord_addr[coord_addr.shape[1] * 2], DOWN_FACTOR,
             coord_addr.shape[1] // BURST_PROPOSAL_NUM, 1, 1, 8, 8)  # x2*DOWN_FACTOR
    ib.vmuls(128, coord_addr[coord_addr.shape[1] * 3], coord_addr[coord_addr.shape[1] * 3], DOWN_FACTOR,
             coord_addr.shape[1] // BURST_PROPOSAL_NUM, 1, 1, 8, 8)  # y2*DOWN_FACTOR

    ib.vadds(128, coord_addr[0], coord_addr[0], 1.0, coord_addr.shape[1] // BURST_PROPOSAL_NUM, 1, 1, 8,
             8)  # x1*DOWN_FACTOR+1
    ib.vadds(128, coord_addr[coord_addr.shape[1] * 1], coord_addr[coord_addr.shape[1] * 1], 1.0,
             coord_addr.shape[1] // BURST_PROPOSAL_NUM, 1, 1, 8, 8)  # y1*DOWN_FACTOR+1

    ib.vconcat(out_proposal[0], coord_addr[0], coord_addr.shape[1] // RPN_PROPOSAL_NUM, 0)  # x1
    ib.vconcat(out_proposal[0], coord_addr[coord_addr.shape[1] * 1], coord_addr.shape[1] // RPN_PROPOSAL_NUM, 1)  # y1
    ib.vconcat(out_proposal[0], coord_addr[coord_addr.shape[1] * 2], coord_addr.shape[1] // RPN_PROPOSAL_NUM, 2)  # x2
    ib.vconcat(out_proposal[0], coord_addr[coord_addr.shape[1] * 3], coord_addr.shape[1] // RPN_PROPOSAL_NUM, 3)  # y2
    return out_proposal


def _tik_func_nms_single_core_multithread(input_shape, thresh, total_output_proposal_num, kernel_name_var):
    """
    Compute output boxes after non-maximum suppression.

    Parameters
    ----------
    input_shape: dict
        shape of input boxes, including proposal boxes and corresponding
    confidence scores

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
    proposals = tik_instance.Tensor("float16", (total_input_proposal_num, 8), name="in_proposals", scope=tik.scope_gm)
    # output shape is [N,5]
    input_ceil = _ceil_div(total_input_proposal_num * 5, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM
    ret = tik_instance.Tensor("float16", (_ceil_div(input_ceil, 5), 5), name="out_proposals", scope=tik.scope_gm)
    # address is 32B aligned
    out_index = tik_instance.Tensor("int32", [_ceil_div(total_output_proposal_num, 8) * 8],
                                    name="out_index",
                                    scope=tik.scope_gm)
    out_mask = tik_instance.Tensor("uint8", [_ceil_div(total_output_proposal_num, 32) * 32],
                                   name="out_mask",
                                   scope=tik.scope_gm)
    output_index_ub = tik_instance.Tensor("int32", [_ceil_div(BURST_PROPOSAL_NUM, 16) * 16],
                                          name="output_index_ub",
                                          scope=tik.scope_ubuf)
    output_mask_ub = tik_instance.Tensor("uint8", [_ceil_div(BURST_PROPOSAL_NUM, 16) * 16],
                                         name="output_mask_ub",
                                         scope=tik.scope_ubuf)
    # variables
    selected_proposals_cnt = tik_instance.Scalar(dtype="uint16")
    selected_proposals_cnt.set_as(0)
    handling_proposals_cnt = tik_instance.Scalar(dtype="uint16")
    handling_proposals_cnt.set_as(0)
    left_proposal_cnt = tik_instance.Scalar(dtype="uint16")
    left_proposal_cnt.set_as(total_input_proposal_num)
    # store the whole
    output_proposals_ub = tik_instance.Tensor("float16",
                                              [_ceil_div(BURST_PROPOSAL_NUM, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM, 5],
                                              name="output_proposals_ub",
                                              scope=tik.scope_ubuf)
    selected_reduced_proposals_ub = tik_instance.Tensor(
        "float16", [_ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM, 8],
        name="selected_reduced_proposals_ub",
        scope=tik.scope_ubuf)
    selected_area_ub = tik_instance.Tensor("float16",
                                           [_ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM],
                                           name="selected_area_ub",
                                           scope=tik.scope_ubuf)
    sup_vec_ub = tik_instance.Tensor("uint16",
                                     [_ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM],
                                     name="sup_vec_ub",
                                     scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, sup_vec_ub[0], 1, 1, 1, 8)
    # change with burst
    temp_reduced_proposals_ub = tik_instance.Tensor("float16", [BURST_PROPOSAL_NUM, 8],
                                                    name="temp_reduced_proposals_ub",
                                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, temp_reduced_proposals_ub[0], 0, 8, 1, 8)
    temp_area_ub = tik_instance.Tensor("float16", [BURST_PROPOSAL_NUM], name="temp_area_ub", scope=tik.scope_ubuf)
    temp_iou_ub = tik_instance.Tensor("float16",
                                      [_ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM, 16],
                                      name="temp_iou_ub",
                                      scope=tik.scope_ubuf)
    temp_join_ub = tik_instance.Tensor("float16",
                                       [_ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM, 16],
                                       name="temp_join_ub",
                                       scope=tik.scope_ubuf)
    temp_sup_matrix_ub = tik_instance.Tensor(
        "uint16", [_ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM],
        name="temp_sup_matrix_ub",
        scope=tik.scope_ubuf)
    temp_sup_vec_ub = tik_instance.Tensor("uint16", [BURST_PROPOSAL_NUM], name="temp_sup_vec_ub", scope=tik.scope_ubuf)
    # main body
    t = tik_instance.Scalar(dtype="uint16")
    t.set_as(0)
    sup_vec_ub[0].set_as(t)
    tt = tik_instance.Scalar(dtype="float16")
    mask = tik_instance.Scalar(dtype="uint8")
    with tik_instance.for_range(0, _ceil_div(total_input_proposal_num, BURST_PROPOSAL_NUM),
                                thread_num=1) as burst_index:
        fresh_proposals_ub = tik_instance.Tensor(
            "float16", [_ceil_div(BURST_PROPOSAL_NUM, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM, 8],
            name="fresh_proposals_ub",
            scope=tik.scope_ubuf)
        # update counter
        with tik_instance.if_scope(left_proposal_cnt < BURST_PROPOSAL_NUM):
            handling_proposals_cnt.set_as(left_proposal_cnt)
        with tik_instance.else_scope():
            handling_proposals_cnt.set_as(BURST_PROPOSAL_NUM)

        # fresh proposals shape is [BURST_PROPOSAL_NUM, 8]
        tik_instance.data_move(fresh_proposals_ub[0], proposals[burst_index * BURST_PROPOSAL_NUM * 8], 0, 1,
                               _ceil_div(handling_proposals_cnt * 16, 32), 0, 0, 0)
        # clear temp_sup_vec_ub
        tik_instance.vector_dup(128, temp_sup_vec_ub[0], 1, temp_sup_vec_ub.shape[0] // BURST_PROPOSAL_NUM, 1, 8)
        # reduce fresh proposal
        _get_reduced_proposal(tik_instance, temp_reduced_proposals_ub, fresh_proposals_ub)
        # calculate the area of reduced-proposal
        tik_instance.vrpac(temp_area_ub[0], temp_reduced_proposals_ub[0], _ceil_div(handling_proposals_cnt, 16))
        # start to update iou and or area from the first 16 proposal
        # and get suppression vector 16 by 16 proposal
        length = tik_instance.Scalar(dtype="uint16")
        length.set_as(_ceil_div(selected_proposals_cnt, 16) * 16)
        with tik_instance.if_scope(selected_proposals_cnt < total_output_proposal_num):
            with tik_instance.new_stmt_scope():
                with tik_instance.for_range(0, _ceil_div(handling_proposals_cnt, 16)) as i:
                    length.set_as(length + 16)
                    # calculate intersection of tempReducedProposals
                    # and selReducedProposals
                    tik_instance.viou(temp_iou_ub[0, 0], selected_reduced_proposals_ub[0],
                                      temp_reduced_proposals_ub[i * 16, 0], _ceil_div(selected_proposals_cnt, 16))
                    # calculate intersection of tempReducedProposals and
                    # tempReducedProposals(include itself)
                    tik_instance.viou(temp_iou_ub[_ceil_div(selected_proposals_cnt, 16) * 16, 0],
                                      temp_reduced_proposals_ub[0], temp_reduced_proposals_ub[i * 16, 0], i + 1)
                    # calculate join of tempReducedProposals
                    # and selReducedProposals
                    tik_instance.vaadd(temp_join_ub[0, 0], selected_area_ub[0], temp_area_ub[i * 16],
                                       _ceil_div(selected_proposals_cnt, 16))
                    # calculate intersection of tempReducedProposals and
                    # tempReducedProposals(include itself)
                    tik_instance.vaadd(temp_join_ub[_ceil_div(selected_proposals_cnt, 16) * 16, 0], temp_area_ub,
                                       temp_area_ub[i * 16], i + 1)
                    # calculate join*(thresh/(1+thresh))
                    tik_instance.vmuls(128, temp_join_ub[0, 0], temp_join_ub[0, 0], thresh, _ceil_div(length, 8), 1, 1,
                                       8, 8)
                    # compare and generate suppression matrix
                    tik_instance.vcmpv_gt(temp_sup_matrix_ub[0], temp_iou_ub[0, 0], temp_join_ub[0, 0],
                                          _ceil_div(length, 8), 1, 1, 8, 8)
                    # generate suppression vector
                    # clear rpn_cor_ir
                    rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)
                    # non-diagonal
                    rpn_cor_ir = tik_instance.rpn_cor(temp_sup_matrix_ub[0], sup_vec_ub[0], 1, 1,
                                                      _ceil_div(selected_proposals_cnt, 16))
                    with tik_instance.if_scope(i > 0):
                        rpn_cor_ir = tik_instance.rpn_cor(
                            temp_sup_matrix_ub[_ceil_div(selected_proposals_cnt, 16) * 16], temp_sup_vec_ub[0], 1, 1, i)
                    # diagonal
                    tik_instance.rpn_cor_diag(temp_sup_vec_ub[i * 16], temp_sup_matrix_ub[length - 16], rpn_cor_ir)

                # find & mov unsuppressed proposals
                with tik_instance.for_range(0, handling_proposals_cnt) as i:
                    with tik_instance.if_scope(selected_proposals_cnt < total_output_proposal_num):
                        with tik_instance.for_range(0, 5) as j:
                            # update selOriginalProposals_ub
                            tt.set_as(fresh_proposals_ub[i, j])
                            output_proposals_ub[i, j].set_as(tt)
                        output_index_ub[i].set_as(i + burst_index * BURST_PROPOSAL_NUM)
                        t.set_as(temp_sup_vec_ub[i])
                        with tik_instance.if_scope(t == 0):
                            with tik_instance.for_range(0, 8) as j:
                                # update selected_reduced_proposals_ub
                                tt.set_as(temp_reduced_proposals_ub[i, j])
                                selected_reduced_proposals_ub[selected_proposals_cnt, j].set_as(tt)
                            # update selected_area_ub
                            tt.set_as(temp_area_ub[i])
                            selected_area_ub[selected_proposals_cnt].set_as(tt)
                            # update sup_vec_ub
                            t.set_as(0)
                            sup_vec_ub[selected_proposals_cnt].set_as(t)
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
                               _ceil_div(handling_proposals_cnt * 5, 16), 0, 0, 0)
        tik_instance.data_move(out_index[burst_index * BURST_PROPOSAL_NUM], output_index_ub, 0, 1,
                               _ceil_div(handling_proposals_cnt, 8), 0, 0, 0)
        tik_instance.data_move(out_mask[burst_index * BURST_PROPOSAL_NUM], output_mask_ub, 0, 1,
                               _ceil_div(handling_proposals_cnt, 32), 0, 0, 0)
    tik_instance.BuildCCE(kernel_name=kernel_name_var,
                          inputs=[proposals],
                          outputs=[ret, out_index, out_mask],
                          output_files_path=None,
                          enable_l2=False)
    return tik_instance


# pylint: disable=unused-argument
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

    def _ceil(x):
        """
        Return the least multiple of 16 integer number
        which is greater than or equal to x.
        """
        return ((x + RPN_PROPOSAL_NUM - 1) // RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM

    # Considering the memory space of Unified_Buffer
    # burst_size + selected_proposal_size + temp_proposal_size + fresh_roposal_size <= UB_size
    fp16_size = tbe_platform.get_bit_len("float16") // 8
    int32_size = tbe_platform.get_bit_len("int32") // 8
    uint8_size = tbe_platform.get_bit_len("uint8") // 8
    uint16_size = tbe_platform.get_bit_len("uint16") // 8
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # output shape is [N,5], including x1,y1,x2,y2,scores
    burst_size = _ceil(BURST_PROPOSAL_NUM) * int32_size + _ceil(BURST_PROPOSAL_NUM) * uint8_size + _ceil(
        BURST_PROPOSAL_NUM) * 5 * fp16_size
    # compute shape is [N,8]
    selected_size = _ceil(input_shape[0]) * 8 * fp16_size + _ceil(input_shape[0]) * fp16_size + _ceil(
        input_shape[0]) * uint16_size
    temp_size = BURST_PROPOSAL_NUM * 8 * fp16_size + BURST_PROPOSAL_NUM * fp16_size + _ceil(
        input_shape[0]) * RPN_PROPOSAL_NUM * fp16_size + _ceil(input_shape[0]) * RPN_PROPOSAL_NUM * fp16_size + _ceil(
            input_shape[0]) * uint16_size + BURST_PROPOSAL_NUM * uint16_size
    fresh_size = _ceil(BURST_PROPOSAL_NUM) * 8 * fp16_size
    used_size = burst_size + selected_size + temp_size + fresh_size
    if used_size > ub_size_bytes:
        error_manager_vector.raise_err_check_params_rules(
            kernel_name, "the number of input boxes out of range(%d B)" % ub_size_bytes, "used size", used_size)

    if input_shape[1] != 8:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the 2nd-dim of input boxes must be equal to 8",
                                                          "box_scores.shape", input_shape)

    output_size, _ = input_shape
    iou_thr = iou_thr / (1 + iou_thr)
    _tik_func_nms_single_core_multithread(input_shape, iou_thr, output_size, kernel_name)
