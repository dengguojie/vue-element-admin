#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ReluGradV2", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"}],
         "case_name": "relu_grad_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,8), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,8),"ori_format": "NC1HWC0"}],
         "case_name": "relu_grad_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,2,8), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,8),"ori_format": "NC1HWC0"}],
         "case_name": "relu_grad_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)




if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)