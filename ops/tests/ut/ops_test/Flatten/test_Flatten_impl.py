#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Flatten", None, None)

case1 = {"params": [{"shape": (255,8,33), "dtype": "float32", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    {"shape": (255,8,33), "dtype": "float32", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"}],
         "case_name": "Flatten_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (255,8,33), "dtype": "float16", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    {"shape": (255,8,33), "dtype": "float16", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"}],
         "case_name": "Flatten_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (4, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"}],
         "case_name": "Flatten_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (4, 16, 64), "dtype": "int32", "format": "ND", "ori_shape": (4, 16, 64),"ori_format": "ND"},
                    {"shape": (4, 16, 64), "dtype": "int32", "format": "ND", "ori_shape": (4, 16, 64),"ori_format": "ND"}],
         "case_name": "Flatten_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)



if __name__ == '__main__':
    ut_case.run("Ascend910")
