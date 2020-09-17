#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
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


if __name__ == '__main__':
    ut_case.run("Ascend910")
