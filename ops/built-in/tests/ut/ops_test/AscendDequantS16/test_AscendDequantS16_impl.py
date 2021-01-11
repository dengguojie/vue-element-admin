#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("AscendDequantS16", None, None)

case1 = {"params": [{"shape": (1,1,1,1,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,1,1,16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_dequants16_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,4,4,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (32,),"ori_format": "ND"},
                    None,
                    {"shape": (1,2,4,4,16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    True],
         "case_name": "ascend_dequants16_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,4,4,16,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,2,1,1,16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,4,4,16,16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_dequants16_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
