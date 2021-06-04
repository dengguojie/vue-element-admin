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

SpaceToBatch ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf
ut_case = OpUT("StridedSliceD", None, None)

case1 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"}, #x
                    {"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"},
                    [0, 0, 0], [4, 8, 16], [3, 4, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 512, 32000), "dtype": "int32", "format": "ND", "ori_shape": (1, 512, 32000),"ori_format": "ND"}, #x
                    {"shape": (1, 512, 32000), "dtype": "int32", "format": "ND", "ori_shape": (1, 512, 32000),"ori_format": "ND"},
                    [0, 0, 0], [1, 512, 32000], [1, 16, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3,), "dtype": "int8", "format": "ND", "ori_shape": (3,),"ori_format": "ND"}, #x
                    {"shape": (3,), "dtype": "int8", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    [0], [2], [1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"}, #x
                    {"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"},
                    [0, 0, 0], [4, 8, 16], [3, 4, 1], 0, 0, 0, 0, 2
                    ],
         "case_name": "StridedSliceD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"}, #x
                    {"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    [0, 0, 0], [1, 512, 1024], [1, 512, 1], 5, 5, 0, 0, 0
                    ],
         "case_name": "StridedSliceD_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"}, #x
                    {"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"},
                    [7, 7, 0], [1, 1, 16], [-3, -1, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_6",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"}, #x
                    {"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"},
                    [0, 0, 0], [4, 9, 16], [3, 4, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (1, 512, 2048), "dtype": "int8", "format": "ND", "ori_shape": (1, 512, 2048),"ori_format": "ND"}, #x
                    {"shape": (1, 512, 2048), "dtype": "int8", "format": "ND", "ori_shape": (1, 512, 2048),"ori_format": "ND"},
                    [0, 0, 0], [1, 512, 2048], [1, 16, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_8",
         "expect": RuntimeError,
         "support_expect": True}

case9 = {"params": [{"shape": (15, 38, 38, 3, 85), "dtype": "float32", "format": "ND", "ori_shape": (15, 38, 38, 3, 85),"ori_format": "ND"}, #x
                    {"shape": (15, 38, 38, 3, 80), "dtype": "float32", "format": "ND", "ori_shape": (15, 38, 38, 3, 80),"ori_format": "ND"},
                    [0, 5], [0, 0], [1, 1], 0, 2, 1, 0, 0
                    ],
         "case_name": "StridedSliceD_9",
         "expect":"success",
         "support_expect": True}

case10 = {"params": [{"shape": (1, 120), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 120),"ori_format": "ND"}, #x
                    {"shape": (1, 80), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 80),"ori_format": "ND"},
                    [0, 40], [1, 120], [1, 1], 0, 0, 0, 0, 0
                    ],
         "case_name": "StridedSliceD_10",
         "expect":"success",
         "support_expect": True}

case11 = {"params": [{"shape": (32, 76, 76, 3, 9), "dtype": "float32", "format": "NCHW", "ori_shape": (32, 76, 76, 3, 9),"ori_format": "ND"}, #x
                     {"shape": (32, 76, 76, 3, 4), "dtype": "float32", "format": "NCHW", "ori_shape": (32, 76, 76, 3, 4),"ori_format": "ND"},
                     [0, 0, 0, 0, 5], [32, 76, 76, 3, 9], [1, 1, 1, 1, 1], 0, 0, 0, 0, 0
                     ],
          "case_name": "StridedSliceD_11",
          "expect":"success",
          "support_expect": True}

case12 = {"params": [{"shape": (7, 7, 3, 188, 192, 16), "dtype": "float32", "format": "NDC1HWC0",
                      "ori_shape": (7, 7, 188, 192, 33), "ori_format": "NDHWC"},
                     {"shape": (7, 7, 3, 188, 192, 16), "dtype": "float32", "format": "NDC1HWC0",
                      "ori_shape": (7, 7, 188, 192, 33), "ori_format": "NDHWC"},
                     [1, 1, 3, 4, 0], [5, 5, 77, 433, -1], [1, 2, 1, 1, 1], 1, 2, 4, 0, 0],
          "case_name": "StridedSliceD_12",
          "expect": "success",
          "support_expect": True}

case13 = {"params": [{"shape": (7, 7, 3, 188, 192, 16), "dtype": "float16", "format": "NDC1HWC0",
                      "ori_shape": (7, 7, 188, 192, 33), "ori_format": "NDHWC"},
                     {"shape": (7, 7, 3, 188, 192, 16), "dtype": "float16", "format": "NDC1HWC0",
                      "ori_shape": (7, 7, 188, 192, 33), "ori_format": "NDHWC"},
                     [1, 1, 3, 4, 16], [5, 5, 77, 43, 33], [1, 2, 1, 1, 1], 1, 2, 4, 0, 0],
          "case_name": "StridedSliceD_13",
          "expect": "success",
          "support_expect": True}

case14 = {"params": [{"shape": (7, 7, 3, 188, 192, 16), "dtype": "float16", "format": "NDC1HWC0",
                      "ori_shape": (7, 7, 188, 192, 33), "ori_format": "NDHWC"},
                     {"shape": (7, 7, 3, 188, 192, 16), "dtype": "float16", "format": "NDC1HWC0",
                      "ori_shape": (7, 7, 188, 192, 33), "ori_format": "NDHWC"},
                     [1, 1, 3, 4, 13], [5, 5, 77, 43, 33], [1, 2, 1, 1, 1], 1, 2, 4, 0, 0],
          "case_name": "StridedSliceD_14",
          "expect": RuntimeError,
          "support_expect": True}

case15 = {"params": [{"shape": (7, 7, 3, 88, 77, 16), "dtype": "int32", "format": "NDC1HWC0",
                      "ori_shape": (7, 33, 7, 88, 77), "ori_format": "NCDHW"},
                     {"shape": (7, 7, 3, 88, 77, 16), "dtype": "int32", "format": "NDC1HWC0",
                      "ori_shape": (7, 33, 7, 88, 77), "ori_format": "NCDHW"},
                     [1, 16, 3, 4, 1], [5, 33, 7, 43, 9], [1, 1, 2, 1, 1], 1, 2, 4, 0, 0],
          "case_name": "StridedSliceD_15",
          "expect": "success",
          "support_expect": True}

case16 = {"params": [{"shape": (16, 1, 1, 1, 64, 2), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 1, 1, 1, 64, 2), "ori_format": "ND"},
                     {"shape": (16, 1, 1, 1, 64, 1), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 1, 1, 1, 64, 1), "ori_format": "ND"},
                     [0, 0], [0, 1], [1, 1], 0, 0, 1, 0, 2],
          "case_name": "StridedSliceD_16",
          "expect": "success",
          "support_expect": True}

case17 = {"params": [{"shape": (32, 1, 1, 1, 64, 2), "dtype": "float32", "format": "ND",
                      "ori_shape": (32, 1, 1, 1, 64, 2), "ori_format": "ND"},
                     {"shape": (32, 1, 1, 1, 64, 1), "dtype": "float32", "format": "ND",
                      "ori_shape": (32, 1, 1, 1, 64, 1), "ori_format": "ND"},
                     [0, 0], [0, 1], [1, 1], 0, 0, 1, 0, 2],
          "case_name": "StridedSliceD_17",
          "expect": "success",
          "support_expect": True}

case18 = {"params": [{"shape": (49, 1, 1, 1, 64, 2), "dtype": "float32", "format": "ND",
                      "ori_shape": (49, 1, 1, 1, 64, 2), "ori_format": "ND"},
                     {"shape": (49, 1, 1, 1, 64, 1), "dtype": "float32", "format": "ND",
                      "ori_shape": (49, 1, 1, 1, 64, 1), "ori_format": "ND"},
                     [0, 0], [0, 1], [1, 1], 0, 0, 1, 0, 2],
          "case_name": "StridedSliceD_18",
          "expect": "success",
          "support_expect": True}

case19 = {"params": [{"shape": (16000, 3121), "dtype": "float16", "format": "ND",
                      "ori_shape": (16000, 3121), "ori_format": "ND"},
                     {"shape": (16000, 1), "dtype": "float16", "format": "ND",
                      "ori_shape": (16000, 1), "ori_format": "ND"},
                     [0, 3120], [16000, 1], [1, 1], 0, 0, 0, 0, 0],
          "case_name": "StridedSliceD_19",
          "expect": "success",
          "support_expect": True}

case20 = {"params": [{"shape": (1, 3, 1344, 2028), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 3, 1344, 2028), "ori_format": "ND"},
                     {"shape": (1, 3, 1344, 6), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 3, 1344, 6), "ori_format": "ND"},
                     [0, 0, 0, 0], [1, 3, 1344, 6], [1, 1, 1, 1], 0, 0, 0, 0, 0],
          "case_name": "StridedSliceD_20",
          "expect": "success",
          "support_expect": True}

case21 = {"params": [{"shape": (32, 3, 16, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (32, 3, 16, 16), "ori_format": "ND"},
                     {"shape": (32, 3, 16, 15), "dtype": "int32", "format": "ND",
                      "ori_shape": (32, 3, 16, 15), "ori_format": "ND"},
                     [0, 0, 0, 0], [32, 3, 16, 15], [1, 1, 1, 1], 0, 0, 0, 0, 0],
          "case_name": "StridedSliceD_21",
          "expect": "success",
          "support_expect": True}


def test_op_select_format(test_arg):
    from impl.strided_slice_d import op_select_format
    op_select_format(
        {"shape": (20, 28, 16, 16, 33), "dtype": "float16", "format": "NDHWC", "ori_shape": (20, 28, 16, 16, 33),
         "ori_format": "NDHWC"},
        {"shape": (20, 28, 16, 16, 33), "dtype": "float16", "format": "NDHWC", "ori_shape": (20, 28, 16, 16, 33),
         "ori_format": "NDHWC"},
        [1, 2, 3, 4, 0], [3, 13, 12, 6, 32], [1, 1, 2, 1, 1], 1, 2, 4, 0, 0)
    op_select_format(
        {"shape": (7, 33, 7, 188, 192), "dtype": "float32", "format": "NCDHW", "ori_shape": (7, 33, 7, 188, 192),
         "ori_format": "NCDHW"},
        {"shape": (7, 33, 7, 188, 192), "dtype": "float32", "format": "NCDHW", "ori_shape": (7, 33, 7, 188, 192),
         "ori_format": "NCDHW"},
        [1, 16, 3, 4, 0], [3, 33, 12, 6, 32], [1, 1, 2, 1, 1], 0, 0, 0, 0, 4)
    op_select_format(
        {"shape": (7, 33, 7, 188, 192), "dtype": "int32", "format": "NCDHW", "ori_shape": (7, 33, 7, 188, 192),
         "ori_format": "NCDHW"},
        {"shape": (7, 33, 7, 188, 192), "dtype": "int32", "format": "NCDHW", "ori_shape": (7, 33, 7, 188, 192),
         "ori_format": "NCDHW"},
        [1, 16, 3, 4, 0], [3, 33, 12, 6, 32], [1, 1, 2, 1, 1], 1, 4, 2, 0, 0)
    op_select_format({"shape": (1, 120), "dtype": "float32", "format": "ND", "ori_shape": (1, 120), "ori_format": "ND"},
                     {"shape": (1, 80), "dtype": "float32", "format": "ND", "ori_shape": (1, 80), "ori_format": "ND"},
                     [0, 40], [1, 120], [1, 1], 0, 0, 0, 0, 0)


ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case8)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case9)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case10)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case11)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case12)
ut_case.add_case(["all"], case13)
ut_case.add_case(["all"], case14)
ut_case.add_case(["all"], case15)
ut_case.add_case(["all"], case16)
ut_case.add_case(["all"], case17)
ut_case.add_case(["all"], case18)
ut_case.add_case(["all"], case19)
ut_case.add_case(["all"], case20)
ut_case.add_case(["all"], case21)
ut_case.add_cust_test_func(test_func=test_op_select_format)

def calc_expect_func(x, y, begin, end, strides):
    inputArr = x['value']
    output = tf.strided_slice(inputArr,np.array(begin),np.array(end),np.array(strides),begin_mask=0,end_mask=0,shrink_axis_mask=0)
    with tf.Session() as sess:
        outputArr = sess.run(output)
    return outputArr

precision_case1 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND","param_type":"input"},
                              {"shape": (2, 2, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 2, 16),"ori_format": "ND","param_type":"output"},
                              [0, 0, 0], [4, 8, 16], [3, 4, 1]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910", precision_case1)


if __name__ == "__main__":
    ut_case.run("Ascend910A")