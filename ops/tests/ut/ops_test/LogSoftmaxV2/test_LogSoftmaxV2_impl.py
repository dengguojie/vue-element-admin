#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("LogSoftmaxV2", None, None)

case1 = {"params": [{"shape": (10,10), "dtype": "float16", "format": "ND", "ori_shape": (10,10),"ori_format": "ND"},
                    {"shape": (10,10), "dtype": "float16", "format": "ND", "ori_shape": (10,10),"ori_format": "ND"},
                    -1],
         "case_name": "LogSoftmaxV2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (5, 9, 6, 11), "dtype": "float32", "format": "ND", "ori_shape": (5, 9, 6, 11),"ori_format": "ND"},
                    {"shape": (5, 9, 6, 11), "dtype": "float32", "format": "ND", "ori_shape": (5, 9, 6, 11),"ori_format": "ND"},
                    (2,3)],
         "case_name": "LogSoftmaxV2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (10, 100, 1000), "dtype": "float16", "format": "ND", "ori_shape": (10, 100, 1000),"ori_format": "ND"},
                    {"shape": (10, 100, 1000), "dtype": "float16", "format": "ND", "ori_shape": (10, 100, 1000),"ori_format": "ND"},
                    3],
         "case_name": "LogSoftmaxV2_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,1,1), "dtype": "float16", "format": "ND", "ori_shape": (1,1,1),"ori_format": "ND"},
                    {"shape": (1,1,1), "dtype": "float16", "format": "ND", "ori_shape": (1,1,1),"ori_format": "ND"},
                    -1],
         "case_name": "LogSoftmaxV2_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2, 1, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 2),"ori_format": "ND"},
                    {"shape": (2, 1, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 2),"ori_format": "ND"},
                    (1,2)],
         "case_name": "LogSoftmaxV2_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)



if __name__ == '__main__':
    ut_case.run("Ascend910")
