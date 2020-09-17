#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPool", None, None)

case1 = {"params": [{"shape": (1,3,35,49,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 35, 49, 16),"ori_format": "NHWC"},
                    {"shape": (1,3,17,24,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 3, 17, 24, 16),"ori_format": "NHWC"},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID"],
         "case_name": "max_pool_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,4,23,111,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 23, 111, 16),"ori_format": "NHWC"},
                    {"shape": (1,4,11,55,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 4, 11, 55, 16),"ori_format": "NHWC"},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "VALID"],
         "case_name": "max_pool_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
         
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)



if __name__ == '__main__':
    ut_case.run("Ascend910")
