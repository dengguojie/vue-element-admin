#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("EluGrad", None, None)

def gen_elu_grad_case(shape_gradient, shape_activation, dtype, case_name_val):
    return {"params": [{"shape": shape_gradient, "dtype": dtype, "ori_shape": shape_gradient, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_activation, "dtype": dtype, "ori_shape": shape_activation, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_gradient, "dtype": dtype, "ori_shape": shape_gradient, "ori_format": "ND", "format": "ND"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_elu_grad_case((32, 112, 15, 112),(32, 112, 15, 112),"float32", "elu_grad_1")
case2 = gen_elu_grad_case((32, 112, 15),(32, 112, 15),"float32", "elu_grad_2")
case3 = gen_elu_grad_case((32, 112),(32, 112),"float32", "elu_grad_3")
case4 = gen_elu_grad_case((32, 112),(32, 112),"float16", "elu_grad_4")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend710", "Ascend910"], case4)

# precision cases
def calc_expect_func(x1, x2, y):
    inputgrad_Arr = x1['value']
    input_Arr = x2['value']
    multiply_Arr = np.multiply(inputgrad_Arr, input_Arr + 1)
    result = np.where(input_Arr > 0, inputgrad_Arr, multiply_Arr)
    return result

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (32, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 16),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })

