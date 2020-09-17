#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("InplaceUpdateD", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    [1]],
         "case_name": "InplaceUpdateD_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                    {"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                    {"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                    [16]],
         "case_name": "InplaceUpdateD_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (1, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (1, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 16),"ori_format": "ND"},
                    [32]],
         "case_name": "InplaceUpdateD_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (2, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (2, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 4, 16),"ori_format": "ND"},
                    [0,1]],
         "case_name": "InplaceUpdateD_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    [1]],
         "case_name": "InplaceUpdateD_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
