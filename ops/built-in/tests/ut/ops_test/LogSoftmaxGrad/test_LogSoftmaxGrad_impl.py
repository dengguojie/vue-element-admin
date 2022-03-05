#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as NP
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
case6 = {"params": [{"shape": (8, 81, 25276), "dtype": "float32", "format": "ND", "ori_shape": (8, 81, 25276),"ori_format": "ND"},
                    {"shape": (8, 81, 25276), "dtype": "float32", "format": "ND", "ori_shape": (8, 81, 25276),"ori_format": "ND"},
                    {"shape": (8, 81, 25276), "dtype": "float32", "format": "ND", "ori_shape": (8, 81, 25276),"ori_format": "ND"}],
         "case_name": "log_softmax_grad_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend710", "Ascend910A"], case6)
def calc_expect_func(inputA, inputB, output):
    axis = -1
    input_A_Arr = inputA['value']
    input_B_Arr = inputB['value']

    exp = NP.exp(input_B_Arr)
    exp_sum = NP.sum(input_A_Arr, axis, keepdims=True)
    exp_sum_broadcast = NP.broadcast_to(exp_sum, inputB['shape'])
    softmax = exp * exp_sum_broadcast
    output_Arr = (input_A_Arr - softmax).astype(output['dtype'])
    return output_Arr

precision_case1 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND","param_type": "input"},
                              {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND","param_type": "input"},
                              {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND","param_type": "output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case2 = {"params": [{"shape": (16, 128), "dtype": "float32", "format": "ND", "ori_shape": (16, 128),"ori_format": "ND","param_type": "input"},
                              {"shape": (16, 128), "dtype": "float32", "format": "ND", "ori_shape": (16, 128),"ori_format": "ND","param_type": "input"},
                              {"shape": (16, 128), "dtype": "float32", "format": "ND", "ori_shape": (16, 128),"ori_format": "ND","param_type": "output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case3 = {"params": [{"shape": (3, 2, 2), "dtype": "float32", "format": "ND", "ori_shape": (3, 2, 2),"ori_format": "ND","param_type": "input"},
                              {"shape": (3, 2, 2), "dtype": "float32", "format": "ND", "ori_shape": (3, 2, 2),"ori_format": "ND","param_type": "input"},
                              {"shape": (3, 2, 2), "dtype": "float32", "format": "ND", "ori_shape": (3, 2, 2),"ori_format": "ND","param_type": "output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case4 = {"params": [{"shape": (3, 2, 2, 3), "dtype": "float32", "format": "ND", "ori_shape": (3, 2, 2, 3),"ori_format": "ND","param_type": "input"},
                              {"shape": (3, 2, 2, 3), "dtype": "float32", "format": "ND", "ori_shape": (3, 2, 2, 3),"ori_format": "ND","param_type": "input"},
                              {"shape": (3, 2, 2, 3), "dtype": "float32", "format": "ND", "ori_shape": (3, 2, 2, 3),"ori_format": "ND","param_type": "output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}

ut_case.add_precision_case("Ascend910",precision_case1)
ut_case.add_precision_case("Ascend910",precision_case2)
ut_case.add_precision_case("Ascend910",precision_case3)
ut_case.add_precision_case("Ascend910",precision_case4)


