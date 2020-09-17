#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("LogSoftmaxGrad", None, None)

case1 = {"params": [{"shape": (1,2), "dtype": "float16", "format": "ND", "ori_shape": (1,2),"ori_format": "ND"},
                    {"shape": (1,2), "dtype": "float16", "format": "ND", "ori_shape": (1,2),"ori_format": "ND"},
                    {"shape": (1,2), "dtype": "float16", "format": "ND", "ori_shape": (1,2),"ori_format": "ND"}],
         "case_name": "log_softmax_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1024,1024), "dtype": "float32", "format": "ND", "ori_shape": (1024,1024),"ori_format": "ND"},
                    {"shape": (1024,1024), "dtype": "float32", "format": "ND", "ori_shape": (1024,1024),"ori_format": "ND"},
                    {"shape": (1024,1024), "dtype": "float32", "format": "ND", "ori_shape": (1024,1024),"ori_format": "ND"},],
         "case_name": "log_softmax_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (3,3), "dtype": "float16", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"},
                    {"shape": (3,3), "dtype": "float16", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"},
                    {"shape": (3,3), "dtype": "float16", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"}],
         "case_name": "log_softmax_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (3,3), "dtype": "float32", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"},
                    {"shape": (3,3), "dtype": "float32", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"},
                    {"shape": (3,3), "dtype": "float32", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"}],
         "case_name": "log_softmax_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (10, 20, 100), "dtype": "float32", "format": "ND", "ori_shape": (10, 20, 100),"ori_format": "ND"},
                    {"shape": (10, 100), "dtype": "float32", "format": "ND", "ori_shape": (10, 100),"ori_format": "ND"},
                    {"shape": (10, 20, 100), "dtype": "float32", "format": "ND", "ori_shape": (10, 20, 100),"ori_format": "ND"}],
         "case_name": "log_softmax_grad_5",
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
    exit(0)
