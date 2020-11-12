#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("SpaceToBatchNdD", None, None)

case1 = {"params": [{"shape": (2,128,48,72,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 48, 72, 2048),"ori_format": "NHWC"},
                    {"shape": (288,128,6,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 6, 8, 2048),"ori_format": "NHWC"},
                    [12, 12],
                    [[12, 12], [12, 12]]],
         "case_name": "space_to_batch_nd_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,128,54,80,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 54, 80, 2048),"ori_format": "NHWC"},
                    {"shape": (338,128,6,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (338, 6, 8, 2048),"ori_format": "NHWC"},
                    [13, 13],
                    [[12, 12], [12, 12]]],
         "case_name": "space_to_batch_nd_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,128,516,72,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 516, 72, 2048),"ori_format": "NHWC"},
                    {"shape": (288,128,6,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 6, 8, 2048),"ori_format": "NHWC"},
                    [12, 12],
                    [[12, 12], [12, 12]]],
         "case_name": "space_to_batch_nd_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)


def calc_expect_func(x, y, block_shape, paddings):
    shape = x['shape']
    inputArr = x['value']
    batch, channel1, height, width, channel0 = shape
    padded_height = height + paddings[0][0] + paddings[0][1]
    padded_width = width + paddings[1][0] + paddings[1][1]
    output_height = padded_height // block_shape[0]
    output_width = padded_width // block_shape[1]
    padded_data = np.pad(inputArr, (
        (0, 0), (0, 0), (paddings[0][0], paddings[0][1]),
        (paddings[1][0], paddings[1][1]), (0, 0)), 'constant')
    tmp1 = padded_data.reshape(
        [batch, channel1, output_height, block_shape[0], output_width,
         block_shape[1], channel0])
    tmp2 = tmp1.transpose((3, 5, 0, 1, 2, 4, 6))
    outputArr = tmp2.reshape(
        [batch * block_shape[0] * block_shape[1], channel1, output_height,
         output_width, channel0])
    print("oushape",outputArr.shape)
    return outputArr

precision_case1 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC","param_type":"input"}, #x
                              {"shape": (2, 1, 4, 4, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (8, 2, 2, 1),"ori_format": "NHWC","param_type":"output"},
                              [1,1], ((1, 1), (1, 1))
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case2 = {"params": [{"shape": (10,2,6,4,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (10,6,4,32),"ori_format": "NHWC","param_type":"input"}, #x
                              {"shape": (40, 2, 5, 4, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (8, 2, 2, 1),"ori_format": "NHWC","param_type":"output"},
                              [2,2], ((1, 3), (1, 3))
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}



ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)


