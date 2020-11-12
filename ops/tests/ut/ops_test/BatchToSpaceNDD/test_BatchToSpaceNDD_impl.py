#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("BatchToSpaceNdD", None, None)

case1 = {"params": [{"shape": (288,128,45,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 45, 8, 2048),"ori_format": "NHWC"},
                    {"shape": (2,128,516,72,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 516, 72, 2048),"ori_format": "NHWC"},
                    [12, 12],
                    [[12, 12], [12, 12]]],
         "case_name": "batch_to_space_nd_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (288,2,6,4000,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 6, 4000, 32),"ori_format": "NHWC"},
                    {"shape": (2,2,48,47976,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 48, 47976, 32),"ori_format": "NHWC"},
                    [12, 12],
                    [[12, 12], [12, 12]]],
         "case_name": "batch_to_space_nd_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (40,2,5,4,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (40,5,4,32),"ori_format": "NHWC"},
                    {"shape": (10,2,6,4,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (10,6,4,32),"ori_format": "NHWC"},
                    [2, 2],
                    [[1, 3], [1, 3]]],
         "case_name": "batch_to_space_nd_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

def calc_expect_func(x, y, block_shape, crops):
    src_type = x['dtype']
    shape = x['shape']
    inputArr = x['value']
    batch, channel1, height, width, channel0 = shape
    padded_height = height * block_shape[0]
    padded_width = width * block_shape[1]
    output_height = padded_height - crops[0][0] - crops[0][1]
    output_width = padded_width - crops[1][0] - crops[1][1]
    tmp1 = inputArr.reshape([block_shape[0], block_shape[1],
                             batch // block_shape[0] // block_shape[1],
                             channel1,
                             height, width, channel0])
    tmp2 = tmp1.transpose(2, 3, 4, 0, 5, 1, 6)
    tmp3 = tmp2.reshape(
        [batch // block_shape[0] // block_shape[1], channel1, padded_height,
         padded_width, channel0])
    tmp4 = tmp3[:, :, crops[0][0]:(padded_height - crops[0][1]), :, :]
    outputArr = tmp4[:, :, :, crops[1][0]:(padded_width - crops[1][1]), :]

    out_shape = [batch // block_shape[0] // block_shape[1], channel1,
                 output_height, output_width, channel0]
    return outputArr

ut_case.add_precision_case("all", {"params": [{"shape": (40,2,5,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (40,2,5,4,16),"ori_format": "NHWC", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (10,2,6,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10,2,6,4,16),"ori_format": "NHWC", "param_type": "output"},
                                              [2,2],[[1, 3], [1, 3]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
# ut_case.add_precision_case("all", {"params": [{"shape": (288,128,45,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 45, 8, 2048),"ori_format": "NCHW", "param_type": "input","value_range":[-10,10]},
#                                               {"shape": (2,128,516,72,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 516, 72, 2048),"ori_format": "NCHW", "param_type": "output"},
#                                               [12,12],[[12, 12], [12, 12]]],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })
