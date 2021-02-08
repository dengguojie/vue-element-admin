#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("BnInferGrad", None, None)

case1 = {"params": [{"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 1, 1, 1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 1, 1, 1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),"ori_format": "NC1HWC0"}],
         "case_name": "BnInferGrad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, 1, 1, 1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 1, 1, 1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 1, 1, 1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 1, 1, 1, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),"ori_format": "NC1HWC0"}],
         "case_name": "BnInferGrad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,1, 1, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1, 1,1, 1, 1, 16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1, 1, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1, 1, 1, 1, 16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1, 1, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1, 1, 1, 1, 16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1, 1, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1, 1, 1, 1, 16),"ori_format": "NDC1HWC0"}],
         "case_name": "BnInferGrad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}         
ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
#ut_case.add_case(["Ascend910A"], case3)

if __name__ == "__main__":
    # ut_case.run()
    ut_case.run("Ascend910A")
    exit(0)