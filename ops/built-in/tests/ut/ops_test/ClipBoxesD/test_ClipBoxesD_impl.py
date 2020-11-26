#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("ClipBoxesD", None, None)

def calc_expect_func(x, y, img):
    img_w = img[0]
    img_h = img[1]
    dataA = x['value']
    compute_input = x['value']
    for i0 in range(x['shape'][0]):
        compute_input[i0, 0] = max(min(dataA[i0, 0], img_w), 0)
        compute_input[i0, 1] = max(min(dataA[i0, 1], img_h), 0)
        compute_input[i0, 2] = max(min(dataA[i0, 2], img_w), 0)
        compute_input[i0, 3] = max(min(dataA[i0, 3], img_h), 0)
    outputArr = compute_input
    return outputArr

case1 = {"params": [{"shape": (4000, 4), "dtype": "float16", "format": "ND", "ori_shape": (4000, 4),"ori_format": "ND"},
                    {"shape": (4000, 4), "dtype": "float16", "format": "ND", "ori_shape": (4000, 4),"ori_format": "ND"},
                    [1024, 728]],
         "case_name": "clip_boxes_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (6000, 4), "dtype": "float16", "format": "ND", "ori_shape": (6000, 4),"ori_format": "ND"},
                    {"shape": (6000, 4), "dtype": "float16", "format": "ND", "ori_shape": (6000, 4),"ori_format": "ND"},
                    [1024, 728]],
         "case_name": "clip_boxes_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (17600, 4), "dtype": "float16", "format": "ND", "ori_shape": (17600, 4),"ori_format": "ND"},
                    {"shape": (17600, 4), "dtype": "float16", "format": "ND", "ori_shape": (17600, 4),"ori_format": "ND"},
                    [1024, 728]],
         "case_name": "clip_boxes_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (4000, 4), "dtype": "float32", "format": "ND", "ori_shape": (4000, 4),"ori_format": "ND"},
                    {"shape": (4000, 4), "dtype": "float32", "format": "ND", "ori_shape": (4000, 4),"ori_format": "ND"},
                    [1024, 728]],
         "case_name": "clip_boxes_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (128,3), "dtype": "float16", "format": "ND", "ori_shape": (128,3),"ori_format": "ND"},
                    {"shape": (128,3), "dtype": "float16", "format": "ND", "ori_shape": (128,3),"ori_format": "ND"},
                    [128, 256]],
         "case_name": "clip_boxes_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

precision_case1 = {"params": [{"shape": (16,4), "dtype": "float16", "format": "ND", "ori_shape": (16,4),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,4), "dtype": "float16", "format": "ND", "ori_shape": (16,4),"ori_format": "ND","param_type":"output"},
                              [1024, 1824]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case2 = {"params": [{"shape": (96,4), "dtype": "float16", "format": "ND", "ori_shape": (96,4),"ori_format": "ND","param_type":"input"},
                              {"shape": (96,4), "dtype": "float16", "format": "ND", "ori_shape": (96,4),"ori_format": "ND","param_type":"output"},
                              [1024, 1824]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case3 = {"params": [{"shape": (6000,4), "dtype": "float16", "format": "ND", "ori_shape": (6000,4),"ori_format": "ND","param_type":"input"},
                              {"shape": (6000,4), "dtype": "float16", "format": "ND", "ori_shape": (6000,4),"ori_format": "ND","param_type":"output"},
                              [1024, 1824]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case4 = {"params": [{"shape": (1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,4),"ori_format": "ND","param_type":"input"},
                              {"shape": (1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,4),"ori_format": "ND","param_type":"output"},
                              [1024, 1824]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case5 = {"params": [{"shape": (4096,4), "dtype": "float16", "format": "ND", "ori_shape": (4096,4),"ori_format": "ND","param_type":"input"},
                              {"shape": (4096,4), "dtype": "float16", "format": "ND", "ori_shape": (4096,4),"ori_format": "ND","param_type":"output"},
                              [1024, 1824]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_precision_case("Ascend910",precision_case1)
ut_case.add_precision_case("Ascend910",precision_case2)
ut_case.add_precision_case("Ascend910",precision_case3)
ut_case.add_precision_case("Ascend910",precision_case4)
ut_case.add_precision_case("Ascend910",precision_case5)







