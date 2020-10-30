#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("RpnProposalsD", None, None)

def gen_data(shape_input, shape_output, dtype,nms_threshold, img_height, img_width,
             score_filter, score_threshold,
             k, box_filter, min_height, min_width,
             score_sigmoid=False, case_name_val=""):

    rois = {"shape": (shape_input, 4), "dtype": dtype, "format":"ND", "ori_shape":(shape_input, 4), "ori_format":"ND"}
    cls_bg_prob = {"shape": (shape_input, 1), "dtype": dtype, "format":"ND", "ori_shape":(shape_input, 1), "ori_format":"ND"}
    sorted_box = {"shape": (shape_output, 4), "dtype": dtype, "format":"ND", "ori_shape":(shape_output, 4), "ori_format":"ND"}

    img_size = (img_height, img_width)
    min_size = min(min_height, min_width)
    post_nms_num = shape_output

    return {"params": [rois, cls_bg_prob, sorted_box, img_size, score_threshold,
                       k, min_size, nms_threshold, post_nms_num, score_filter,
                       box_filter, score_sigmoid],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

# runtime too long
case1 = gen_data(501120, 96, "float16",0.7, 1856, 2880, True, 0, 6000, True, 0, 0, False,
                 "rpn_proposals_d_1")
case2 = gen_data(72000, 16, "float16",0.7, 720, 1280, True, 0, 6000, True, 0, 0, False,
                 "rpn_proposals_d_2")
case3 = gen_data(7200, 16, "float16",0.7, 720, 1280, True, 0, 0, True, 0, 0, False,
                 "rpn_proposals_d_3")

# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)

