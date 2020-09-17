#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("InplaceAddD", None, None)

case1 = {"params": [{"shape": (4,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (4,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (4,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (4,4,32,2),"ori_format": "ND"},
                    [0, 1]],
         "case_name": "inplace_add_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (4,4,32), "dtype": "float32", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    [0, 1]],
         "case_name": "inplace_add_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    [0, 1]],
         "case_name": "inplace_add_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,4,32,3), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,3),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,3), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,3),"ori_format": "ND"},
                    [0, 1]],
         "case_name": "inplace_add_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)