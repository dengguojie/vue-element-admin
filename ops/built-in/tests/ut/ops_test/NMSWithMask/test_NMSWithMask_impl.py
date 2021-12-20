# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

NMSWithMask ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
from impl.nms_with_mask import nms_with_mask
from te import platform as cce_conf

ut_case = OpUT("NMSWithMask", "impl.nms_with_mask", "nms_with_mask")

case_small_shape_not_aligned = {
    "params":
        [
            {
                "shape": (6, 8),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (6, 8),
                "ori_format": "ND"
            },
            {
                "shape": (6, 5),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (6, 5),
                "ori_format": "ND"
            },
            {
                "shape": (6,),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (6,),
                "ori_format": "ND"
            },
            {
                "shape": (6,),
                "format": "ND",
                "dtype": "uint8",
                "ori_shape": (6,),
                "ori_format": "ND"
            },
            0.7
        ],
    "case_name": 'test_nms_with_mask_small_shape_not_aligned',
    "expect": "success"
}

case_big_shape_not_aligned = {
    "params":
        [
            {
                "shape": (2007, 8),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (2007, 8),
                "ori_format": "ND"
            },
            {
                "shape": (2007, 5),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (2007, 5),
                "ori_format": "ND"
            },
            {
                "shape": (2007,),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (2007,),
                "ori_format": "ND"
            },
            {
                "shape": (2007,),
                "format": "ND",
                "dtype": "uint8",
                "ori_shape": (2007,),
                "ori_format": "ND"
            },
            0.7
        ],
    "case_name": 'test_nms_with_mask_big_shape_not_aligned',
    "expect": "success"
}

case_aligned_with_iou_equal_one = {
    "params":
        [
            {
                "shape": (16, 8),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 8),
                "ori_format": "ND"
            },
            {
                "shape": (16, 5),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 5),
                "ori_format": "ND"
            },
            {
                "shape": (16,),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (16,),
                "ori_format": "ND"
            },
            {
                "shape": (16,),
                "format": "ND",
                "dtype": "uint8",
                "ori_shape": (16,),
                "ori_format": "ND"
            },
            1.0
        ],
    "case_name": 'test_nms_with_mask_aligned_with_iou_equal_one',
    "expect": "success"
}

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend920A"], case_small_shape_not_aligned)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend920A"], case_big_shape_not_aligned)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend920A"], case_aligned_with_iou_equal_one)


def nms_proposals_reduce(proposals, downscale):
    """
    down scaling of proposals

    Parameters:
    ----------
    proposals: to be scaled
    downscale: scaling factor

    Returns
    -------
    reduced_proposals
    """
    reduced_proposals = np.zeros(proposals.shape, dtype=np.float16)
    for i in range(proposals.shape[0]):
        for j in range(2):
            reduced_proposals[i, j] = proposals[i, j] * downscale + 1
        for j in range(2, 4):
            reduced_proposals[i, j] = proposals[i, j] * downscale
        reduced_proposals[i, 4:] = proposals[i, 4:]
    return reduced_proposals


def nms_proposals_area(proposals):
    """
    area calculation

    Parameters:
    ----------
    proposals: from proposals calculate areas

    Returns
    -------
    area
    """
    area = np.zeros(proposals.shape[0], dtype=np.float16)
    for i in range(proposals.shape[0]):
        area[i] = (proposals[i][2] - proposals[i][0] + 1) * (proposals[i][3] - proposals[i][1] + 1)
    return area


def nms_proposal_aadd(area_0, area_1, thresh):
    """
    area add

    Parameters:
    ----------
    area_0: area 0
    area_1: area 1
    thresh: threshold

    Returns
    -------
    aadd
    """
    return (area_0 + area_1) * thresh


def nms_proposal_iou(proposal_0, proposal_1):
    """
    intersection calc, which is used for iou

    Parameters:
    ----------
    proposal_0: one proposal
    proposal_1: another proposal

    Returns
    -------
    intersection
    """
    xx1 = max(proposal_0[0], proposal_1[0])
    yy1 = max(proposal_0[1], proposal_1[1])
    xx2 = min(proposal_0[2], proposal_1[2])
    yy2 = min(proposal_0[3], proposal_1[3])
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    return w * h


def calc_expect_func(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr):
    """
    expect function

    Parameters:
    ----------
    box_scores: input boxes and scores
    selected_boxes: output selected boxes
    selected_idx: output selected idx
    selected_mask: output selected mask
    iou_thr: iou threshold

    Returns
    -------
    [boxes_proposals, out_idx, out_mask]
    """
    proposals_in = box_scores["value"]
    downscale = 0.054395
    thresh = iou_thr / (1 + iou_thr)
    input_num, _ = proposals_in.shape
    output_num = input_num
    output_shape = [output_num, 8]
    out_idx = np.zeros(output_num, dtype=selected_idx["dtype"])
    for i in range(input_num):
        out_idx[i] = i  # label

    reduced_proposals_in = nms_proposals_reduce(proposals_in, downscale)
    area = nms_proposals_area(reduced_proposals_in)

    # simulate with python
    supVec = np.zeros(output_num, dtype=selected_idx["dtype"])  # not suppressed
    supVec[0] = 0
    proposals_out = np.zeros(output_shape, dtype=selected_boxes["dtype"])

    for i in range(0, input_num - 1):
        for j in range((i + 1), input_num):
            if (supVec[i] == 0):
                intersec = nms_proposal_iou(reduced_proposals_in[i], reduced_proposals_in[j])
                join = nms_proposal_aadd(area[i], area[j], thresh)
                if (intersec > join):
                    supVec[j] = 1

    out_mask = np.zeros(output_num, dtype=selected_mask["dtype"])
    for i in range(0, input_num):
        if (supVec[i] == 0):
            out_mask[i] = 1
        for j in range(5):
            proposals_out[i, j] = proposals_in[i, j]

    boxes_shape = [output_num, 5]
    boxes_proposals = np.zeros(boxes_shape, dtype=selected_boxes["dtype"])
    for i in range(output_num):
        for j in range(5):
            boxes_proposals[i, j] = proposals_out[i, j]
    return [boxes_proposals.astype(selected_boxes["dtype"]), out_idx.astype(selected_idx["dtype"]),
            out_mask.astype(selected_mask["dtype"])]


def test_a100_fp16_small_shape(test_arg):
    """
    test small shape, dtype is fp16

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    n_proposals = 6
    dtype = 'float16'

    nms_with_mask(
        {
            "shape": (n_proposals, 8),
            "format": "ND",
            "dtype": dtype,
            "ori_shape": (n_proposals, 8),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals, 5),
            "format": "ND",
            "dtype": dtype,
            "ori_shape": (n_proposals, 5),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals,),
            "format": "ND",
            "dtype": "int32",
            "ori_shape": (n_proposals,),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals,),
            "format": "ND",
            "dtype": "uint8",
            "ori_shape": (n_proposals,),
            "ori_format": "ND"
        },
        0.5)

    cce_conf.cce_conf.te_set_version(test_arg)


def test_a100_fp16_big_shape(test_arg):
    """
    test big shape, dtype is fp16

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    n_proposals = 2007
    dtype = 'float16'

    nms_with_mask(
        {
            "shape": (n_proposals, 8),
            "format": "ND",
            "dtype": dtype,
            "ori_shape": (n_proposals, 8),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals, 5),
            "format": "ND",
            "dtype": dtype,
            "ori_shape": (n_proposals, 5),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals,),
            "format": "ND",
            "dtype": "int32",
            "ori_shape": (n_proposals,),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals,),
            "format": "ND",
            "dtype": "uint8",
            "ori_shape": (n_proposals,),
            "ori_format": "ND"
        },
        0.5)

    cce_conf.cce_conf.te_set_version(test_arg)


def test_a100_fp32_small_shape(test_arg):
    """
    test small shape, dtype is fp32

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    n_proposals = 6
    dtype = 'float32'

    nms_with_mask(
        {
            "shape": (n_proposals, 8),
            "format": "ND",
            "dtype": dtype,
            "ori_shape": (n_proposals, 8),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals, 5),
            "format": "ND",
            "dtype": dtype,
            "ori_shape": (n_proposals, 5),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals,),
            "format": "ND",
            "dtype": "int32",
            "ori_shape": (n_proposals,),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals,),
            "format": "ND",
            "dtype": "uint8",
            "ori_shape": (n_proposals,),
            "ori_format": "ND"
        },
        0.5)

    cce_conf.cce_conf.te_set_version(test_arg)


def test_a100_fp32_big_shape(test_arg):
    """
    test big shape, dtype is fp32

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    n_proposals = 1001
    dtype = 'float32'

    nms_with_mask(
        {
            "shape": (n_proposals, 8),
            "format": "ND",
            "dtype": dtype,
            "ori_shape": (n_proposals, 8),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals, 5),
            "format": "ND",
            "dtype": dtype,
            "ori_shape": (n_proposals, 5),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals,),
            "format": "ND",
            "dtype": "int32",
            "ori_shape": (n_proposals,),
            "ori_format": "ND"
        },
        {
            "shape": (n_proposals,),
            "format": "ND",
            "dtype": "uint8",
            "ori_shape": (n_proposals,),
            "ori_format": "ND"
        },
        0.5)

    cce_conf.cce_conf.te_set_version(test_arg)


#ut_case.add_cust_test_func(test_func=test_a100_fp16_small_shape)
#ut_case.add_cust_test_func(test_func=test_a100_fp16_big_shape)
#ut_case.add_cust_test_func(test_func=test_a100_fp32_small_shape)
#ut_case.add_cust_test_func(test_func=test_a100_fp32_big_shape)

ut_case.run(['Ascend920A', 'Ascend910A', 'Ascend310'])
