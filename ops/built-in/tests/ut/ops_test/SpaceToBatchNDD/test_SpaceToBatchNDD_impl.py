#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("SpaceToBatchNdD", None, None)

#NHWC-3D-branch_1
case1 = {"params": [{"shape": (8,2,1,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (8,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,1,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,32),"ori_format": "NHWC"},
                    [2], [[1, 1]]],
         "case_name": "space_to_batch_nd_d_1",
         "expect": "success",
         "support_expect": True}
case1_1 = {"params": [{"shape": (8,2,1,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (8,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,1,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,32),"ori_format": "NHWC"},
                    [2], [1, 1]],
         "case_name": "space_to_batch_nd_d_1_1",
         "expect": "success",
         "support_expect": True}
#NHWC-4D-brach_1
case2 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_2",
         "expect": "success",
         "support_expect": True}
case2_1 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                    [2, 2], [1, 1, 1, 1]],
         "case_name": "space_to_batch_nd_d_2_1",
         "expect": "success",
         "support_expect": True}
case2_2 = {"params": [{"shape": (4,2,2,2,16,2), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,2,2,16,2), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_2_2",
         "expect": RuntimeError,
         "support_expect": True}
case2_3 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                      {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                      [1,2, 2], [[1, 1], [1, 1]]],
           "case_name": "space_to_batch_nd_d_2_3",
           "expect": RuntimeError,
           "support_expect": True}
case2_4 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                    [2, 2], [[0,0],[1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_2_4",
         "expect": RuntimeError,
         "support_expect": True}
case2_5 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                      {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                      [0, 0], [[1, 1], [1, 1]]],
           "case_name": "space_to_batch_nd_d_2_5",
           "expect": RuntimeError,
           "support_expect": True}
case2_6 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                      {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                      [2, 2], [[-1, -1], [-1, -1]]],
           "case_name": "space_to_batch_nd_d_2_6",
           "expect": RuntimeError,
           "support_expect": True}
case2_7 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                      {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                      [2, 2], [[1, 2], [1, 1]]],
           "case_name": "space_to_batch_nd_d_2_7",
           "expect": RuntimeError,
           "support_expect": True}
case2_8 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                      {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                      [2, 2], [[1, 1], [1, 2]]],
           "case_name": "space_to_batch_nd_d_2_8",
           "expect": RuntimeError,
           "support_expect": True}
#NHWC-4D-brach_2
case3 = {"params": [{"shape": (4,2,1998,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,1998,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,1000,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1000,2,32),"ori_format": "NHWC"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_3",
         "expect": "success",
         "support_expect": True}
#NHWC-4D-brach_3
case4 = {"params": [{"shape": (4,2,2,3998,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,3998,32),"ori_format": "NHWC"},
                    {"shape": (16000,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16000,2,2,32),"ori_format": "NHWC"},
                    [2, 2000], [[1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_4",
         "expect": "success",
         "support_expect": True}
#NHWC-4D-brach_4
case5 = {"params": [{"shape": (4,2,2,7998,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,7998,32),"ori_format": "NHWC"},
                    {"shape": (16,2,2,4000,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,4000,32),"ori_format": "NHWC"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_5",
         "expect": "success",
         "support_expect": True}
#NDHWC-5D-brach_1
case6 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                    {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_6",
         "expect": "success",
         "support_expect": True}
case6_1 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                    {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [1, 1, 1, 1, 1, 1]],
         "case_name": "space_to_batch_nd_d_6_1",
         "expect": "success",
         "support_expect": True}
case6_2 = {"params": [{"shape": (2,62,2,2,2,16,2), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                    {"shape": (16,32,2,2,2,16,2), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_6_2",
         "expect": RuntimeError,
         "support_expect": True}
case6_3 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                      {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                      [1,2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
           "case_name": "space_to_batch_nd_d_6_3",
           "expect": RuntimeError,
           "support_expect": True}
case6_4 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                      {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                      [2, 2, 2], [[0,0],[1, 1], [1, 1], [1, 1]]],
           "case_name": "space_to_batch_nd_d_6_4",
           "expect": RuntimeError,
           "support_expect": True}
case6_5 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                      {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                      [0, 0, 0], [[1, 1], [1, 1], [1, 1]]],
           "case_name": "space_to_batch_nd_d_6_5",
           "expect": RuntimeError,
           "support_expect": True}
case6_6 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                      {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                      [2, 2, 2], [[-1, -1], [-1, -1], [-1, -1]]],
           "case_name": "space_to_batch_nd_d_6_6",
           "expect": RuntimeError,
           "support_expect": True}
case6_7 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                      {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                      [2, 2, 2], [[1, 2], [1, 1], [1, 1]]],
           "case_name": "space_to_batch_nd_d_6_7",
           "expect": RuntimeError,
           "support_expect": True}
case6_8 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                      {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                      [2, 2, 2], [[1, 1], [1, 2], [1, 1]]],
           "case_name": "space_to_batch_nd_d_6_8",
           "expect": RuntimeError,
           "support_expect": True}
case6_9 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                      {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC"},
                      [2, 2, 2], [[1, 1], [1, 1], [1, 2]]],
           "case_name": "space_to_batch_nd_d_6_9",
           "expect": RuntimeError,
           "support_expect": True}
#NDHWC-5D-brach_1
case7 = {"params": [{"shape": (2,126,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,126,2,2,32),"ori_format": "NDHWC"},
                    {"shape": (16,64,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,64,2,2,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_7",
         "expect": "success",
         "support_expect": True}
#NDHWC-5D-brach_1
case8 = {"params": [{"shape": (2,62,248,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,3968),"ori_format": "NDHWC"},
                    {"shape": (16,32,248,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,3968),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_8",
         "expect": "success",
         "support_expect": True}
#NDHWC-5D-brach_2
case9 = {"params": [{"shape": (2,62,2,3998,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,3998,2,32),"ori_format": "NDHWC"},
                    {"shape": (16,32,2,2000,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2000,2,32),"ori_format": "NDHWC"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_9",
         "expect": "success",
         "support_expect": True}
#NDHWC-5D-brach_3
case10 = {"params": [{"shape": (2,62,2,2,7998,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,7998,32),"ori_format": "NDHWC"},
                     {"shape": (16,32,2,2,4000,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,4000,32),"ori_format": "NDHWC"},
                     [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
          "case_name": "space_to_batch_nd_d_10",
          "expect": "success",
          "support_expect": True}
#NCHW-4D-brach_1-fp32
case11 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (4,32,2,2),"ori_format": "NCHW"},
                    {"shape": (16,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16,32,2,2),"ori_format": "NCHW"},
                    [1, 2, 2], [[0, 0], [1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_11",
         "expect": "success",
         "support_expect": True}
case11_1 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (4,32,2,2),"ori_format": "NCHW"},
                     {"shape": (16,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16,32,2,2),"ori_format": "NCHW"},
                     [1, 2, 2], [0, 0, 1, 1, 1, 1]],
          "case_name": "space_to_batch_nd_d_11_1",
          "expect": "success",
          "support_expect": True}
#NCDHW-5D-brach_1-fp32
case12 = {"params": [{"shape": (2,62,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,16,62,2,2),"ori_format": "NCDHW"},
                    {"shape": (16,32,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,16,32,2,2),"ori_format": "NCDHW"},
                    [1,2, 2, 2], [[0,0],[1, 1], [1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_12",
         "expect": "success",
         "support_expect": True}
case12_1 = {"params": [{"shape": (2,62,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,16,62,2,2),"ori_format": "NCDHW"},
                     {"shape": (16,32,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,16,32,2,2),"ori_format": "NCDHW"},
                     [1,2, 2, 2], [0,0,1, 1, 1, 1, 1, 1]],
          "case_name": "space_to_batch_nd_d_12_1",
          "expect": "success",
          "support_expect": True}
#format not nc1hwc0 and ndc1hwc0
case13 = {"params": [{"shape": (4,2,2,32), "dtype": "float16", "format": "NHWC", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,2,32), "dtype": "float16", "format": "NHWC", "ori_shape": (16,2,2,32),"ori_format": "NHWC"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_13",
         "expect": RuntimeError,
         "support_expect": True}
#ori_format not nhwc and nchw
case13_1 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "ND"},
                    {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "ND"},
                    [2, 2], [[1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_13_1",
         "expect": RuntimeError,
         "support_expect": True}
#ori_format not ndhwc and ncdhw
case13_2 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "ND"},
                    {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "ND"},
                    [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
         "case_name": "space_to_batch_nd_d_13_2",
         "expect": RuntimeError,
         "support_expect": True}
#NHWC-4D-copy_only
case14 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    {"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    [1, 1], [[0, 0], [0, 0]]],
         "case_name": "space_to_batch_nd_d_14",
         "expect": "success",
         "support_expect": True}
#NHWC-4D-transpose
case15 = {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC"},
                    {"shape": (16,2,4,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,4,4,32),"ori_format": "NHWC"},
                    [2, 2], [[0, 0], [0, 0]]],
         "case_name": "space_to_batch_nd_d_15",
         "expect": "success",
         "support_expect": True}
#NDHWC-5D-copy_only
case16 = {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                    {"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC"},
                    [1, 1, 1], [[0, 0], [0, 0], [0, 0]]],
         "case_name": "space_to_batch_nd_d_16",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1_1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2_1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2_2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2_3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2_4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2_5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2_6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2_7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2_8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6_9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case11_1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case12_1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case13)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case13_1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case13_2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case14)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case15)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case16)

def calc_expect_func_5hd(x, y, block_shape, paddings):
    shape = x['shape']
    inputArr = x['value']
    ori_format = x['ori_format']
    if ori_format == 'NCHW':
        block_shape = [block_shape[1], block_shape[2]]
        paddings = [[paddings[1][0],paddings[1][1]],[paddings[2][0],paddings[2][1]]]
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
    return outputArr

def calc_expect_func_6hd(x, y, block_shape, paddings):
    shape = x['shape']
    inputArr = x['value']
    ori_format = x['ori_format']
    if ori_format == 'NCDHW':
        block_shape = [block_shape[1], block_shape[2], block_shape[3]]
        paddings = [[paddings[1][0],paddings[1][1]],[paddings[2][0],paddings[2][1]],[paddings[3][0],paddings[3][1]]]
    batch, depth, channel1, height, width, channel0 = shape
    padded_depth = depth + paddings[0][0] + paddings[0][1]
    padded_height = height + paddings[1][0] + paddings[1][1]
    padded_width = width + paddings[2][0] + paddings[2][1]
    output_depth = padded_depth // block_shape[0]
    output_height = padded_height // block_shape[1]
    output_width = padded_width // block_shape[2]
    padded_data = np.pad(inputArr, (
        (0, 0), (paddings[0][0], paddings[0][1]), (0, 0), (paddings[1][0], paddings[1][1]),
        (paddings[2][0], paddings[2][1]), (0, 0)), 'constant')
    tmp1 = padded_data.reshape(
        [batch, output_depth, block_shape[0], channel1, output_height, block_shape[1], output_width,
         block_shape[2], channel0])
    tmp2 = tmp1.transpose((2, 5, 7, 0, 1, 3, 4, 6, 8))
    outputArr = tmp2.reshape(
        [batch * block_shape[0] * block_shape[1] * block_shape[2], output_depth, channel1, output_height,
         output_width, channel0])
    return outputArr

#NHWC-4D-brach_1
ut_case.add_precision_case("all", {"params": [{"shape": (4,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,2,2,32),"ori_format": "NHWC", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (16,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,2,32),"ori_format": "NHWC", "param_type": "output"},
                                              [2, 2],[[1, 1], [1, 1]]],
                                   "calc_expect_func": calc_expect_func_5hd,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
#NCHW-4D-brach_1-fp32
ut_case.add_precision_case("all", {"params": [{"shape": (4,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (4,32,2,2),"ori_format": "NCHW", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (16,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16,32,2,2),"ori_format": "NCHW", "param_type": "output"},
                                              [1, 2, 2], [[0, 0], [1, 1], [1, 1]]],
                                   "calc_expect_func": calc_expect_func_5hd,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
#NDHWC-5D-brach_1
ut_case.add_precision_case("all", {"params": [{"shape": (2,62,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,62,2,2,32),"ori_format": "NDHWC", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (16,32,2,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,32,2,2,32),"ori_format": "NDHWC", "param_type": "output"},
                                              [2, 2, 2], [[1, 1], [1, 1], [1, 1]]],
                                   "calc_expect_func": calc_expect_func_6hd,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
#NCDHW-5D-brach_1-fp32
ut_case.add_precision_case("all", {"params": [{"shape": (2,62,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,16,62,2,2),"ori_format": "NCDHW", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (16,32,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16,16,32,2,2),"ori_format": "NCDHW", "param_type": "output"},
                                              [1,2, 2, 2], [[0,0],[1, 1], [1, 1], [1, 1]]],
                                   "calc_expect_func": calc_expect_func_6hd,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })