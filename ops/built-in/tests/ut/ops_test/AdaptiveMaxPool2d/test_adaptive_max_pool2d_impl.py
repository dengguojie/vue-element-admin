#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import torch
import torch.nn as nn
ut_case = OpUT("AdaptiveMaxPool2d", None, None)

case1 = {"params": [{"shape": (1,1,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 16, 5, 5),"ori_format": "NCHW"},
                    {"shape": (1,1,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 16, 5, 5),"ori_format": "NCHW"},
                    [5, 5]],
         "case_name": "adaptive_max_pool2d_case_001",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (11,11,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (11, 176, 5, 5),"ori_format": "NCHW"},
                    {"shape": (11,11,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (11, 176, 3, 3),"ori_format": "NCHW"},
                    [3, 3]],
         "case_name": "adaptive_max_pool2d_case_002",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,1,7,4096,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 112, 7, 4096),"ori_format": "NCHW"},
                    {"shape": (1,1,3,37,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 48, 3, 37),"ori_format": "NCHW"},
                    [3, 37]],
         "case_name": "adaptive_max_pool2d_case_003",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

from impl.adaptive_max_pool2d import check_supported

# pylint: disable=unused-argument,unused-variable
def test_check_support(test_arg):
    # x, y, output_size
    res = check_supported(
                   {"shape": (11,11,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (11, 176, 5, 5),"ori_format": "NCHW"},
                    {"shape": (11,11,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (11, 176, 3, 3),"ori_format": "NCHW"},
                    [3, 3],
                    "adaptive_max_pool2d_check_support_case_001")
    assert res


ut_case.add_cust_test_func(test_func=test_check_support)

#precision cases
def NC1HWC02NCHW(fmi, fmi_shape, precise):
    fmo = np.zeros((fmi_shape[0], fmi_shape[1]*fmi_shape[4], fmi_shape[2], fmi_shape[3]), dtype=np.float16)
    for n in range(fmi_shape[0]):
        for c1 in range(fmi_shape[1]):
            for h in range(fmi_shape[2]):
                for w in range(fmi_shape[3]):
                    for c0 in range(fmi_shape[4]):
                        fmo[n][c1*fmi_shape[4]+c0][h][w] = fmi[n][c1][h][w][c0]
    return fmo

#NCHW2NC1HWC0
def NCHW2NC1HWC0(fmi, fmo_shape, precise):
    fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.float16)
    for n in range(fmo_shape[0]):
        for c1 in range(fmo_shape[1]):
            for h in range(fmo_shape[2]):
                for w in range(fmo_shape[3]):
                    for c0 in range(fmo_shape[4]):
                        fmo[n][c1][h][w][c0] = fmi[n][c1*fmo_shape[4]+c0][h][w]
    return fmo

def calc_expect_func(x, y, output_size):
    inputArr = x['value']
    shape = x['shape']
    inputArr_NCHW = NC1HWC02NCHW(inputArr, shape, "float16")

    m = torch.nn.AdaptiveMaxPool2d(tuple(output_size))
    outputArr_NCHW = m(inputArr_NCHW) 
    #output shape
    batch, channel, height, width = outputArr_NCHW.shape
    C0 = shape[4]
    C1 = (channel + C0 - 1) // C0
    shape_output = [batch, C1, height, width, C0]
    outputArr = NCHW2NC1HWC0(outputArr_NCHW, shape_output, "float16")
    return outputArr

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (11,11,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (11, 176, 5, 5),"ori_format": "NCHW"},
                                                    {"shape": (11,11,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (11, 176, 3, 3),"ori_format": "NCHW"},
                                                    [3, 3]],
                                         "case_name": "adaptive_max_pool2d_prec_case_001",
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})



