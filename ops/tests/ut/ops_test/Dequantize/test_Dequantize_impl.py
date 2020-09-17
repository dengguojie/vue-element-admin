#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("Dequantize", None, None)

case1 = {"params": [{"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    "MIN_COMBINED"],
         "case_name": "dequantize_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    "MIN_FIRST"],
         "case_name": "dequantize_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    "SCALED"],
         "case_name": "dequantize_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run()
    exit(0)