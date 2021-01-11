#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("AscendRequant", None, None)

case1 = {"params": [{"shape": (1,1,1,1,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,1,1,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    True],
         "case_name": "ascend_requant_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,4,4,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (32,),"ori_format": "ND"},
                    {"shape": (1,1,4,4,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4),"ori_format": "ND"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,4),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4),"ori_format": "ND"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,4),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_4",
         "expect": AttributeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2,1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "ND"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (2,1,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_5",
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


