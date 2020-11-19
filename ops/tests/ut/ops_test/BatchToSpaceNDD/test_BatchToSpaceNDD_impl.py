#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("BatchToSpaceNdD", None, None)

#NHWC-4D-brach_1
case1 = {"params": [{"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                    {"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NHWC-4D-brach_2
case2 = {"params": [{"shape": (16,2,1000,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16,1000,2,32),"ori_format": "NHWC"},
                    {"shape": (4,2,1998,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (4,1998,2,32),"ori_format": "NHWC"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NHWC-4D-brach_3
case3 = {"params": [{"shape": (16000,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16000,2,2,32),"ori_format": "NHWC"},
                    {"shape": (4,2,2,3998,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,3998,32),"ori_format": "NHWC"},
                    [2, 2000], [[1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NHWC-4D-brach_4
case4 = {"params": [{"shape": (16,2,2,4000,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16,2,4000,32),"ori_format": "NHWC"},
                    {"shape": (4,2,2,7998,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (4,2,7998,32),"ori_format": "NHWC"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NHWC-3D
case5 = {"params": [{"shape": (16,2,1,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,32),"ori_format": "NHWC"},
                    {"shape": (8,2,1,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (8,2,32),"ori_format": "NHWC"},
                    [2], [[1, 1]]],
         "case_name": "batch_to_space_nd_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NDHWC-5D-brach_1
case6 = {"params": [{"shape": (16,32,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                    {"shape": (2,62,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NDHWC-5D-brach_2
case7 = {"params": [{"shape": (16,64,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,64,2,2,32),"ori_format": "NDHWC"},
                    {"shape": (2,126,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,126,2,2,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NDHWC-5D-brach_3
case8 = {"params": [{"shape": (16,32,248,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,3968),"ori_format": "NDHWC"},
                    {"shape": (2,62,248,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,3968),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NDHWC-5D-brach_4
case9 = {"params": [{"shape": (16,32,2,2600,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,600,2,32),"ori_format": "NDHWC"},
                    {"shape": (2,62,2,1198,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,1198,2,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
#NDHWC-5D-brach_5
case10 = {"params": [{"shape": (16,32,2,2,4000,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16,32,2,4000,32),"ori_format": "NDHWC"},
                    {"shape": (2,62,2,2,7998,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (2,62,2,7998,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "batch_to_space_nd_d_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case10)

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

#NHWC-4D-brach_1
ut_case.add_precision_case("all", {"params": [{"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC", "param_type": "output"},
                                              [2, 2],[[1, 1], [1, 1]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
