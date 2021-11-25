#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("ApplyMomentumD", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "apply_momentum_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "apply_momentum_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"},
                    {"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"},
                    {"shape": (0,), "dtype": "float32", "format": "ND", "ori_shape": (0,),"ori_format": "ND"}],
         "case_name": "apply_momentum_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"},
                    {"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"},
                    {"shape": (0.77,), "dtype": "float32", "format": "ND", "ori_shape": (0.77,),"ori_format": "ND"}],
         "case_name": "apply_momentum_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


def test_apply_get_op_support_info(test_arg):
    from impl.apply_momentum_d import get_op_support_info
    get_op_support_info(
        {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, None)


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_cust_test_func(test_func=test_apply_get_op_support_info)


def calc_expect_func(x1, x2, x3, x4, x5,y1, y2):
    use_nesterov=False
    dtype = x1['dtype']
    s_type =dtype
    input_var = x1['value']
    input_accum = x2['value']
    input_lr = x3['value']
    input_grad = x4['value']
    input_momentum = x5['value']
    if dtype == "float16":
        input_var = input_var.astype(np.float32)
        input_accum = input_accum.astype(np.float32)
        input_lr = input_lr.astype(np.float32)
        input_grad = input_grad.astype(np.float32)
        input_momentum = input_momentum.astype(np.float32)

    # Calc output
    output_accum = input_accum * input_momentum + input_grad

    if use_nesterov is True:
        print("True")
        nesterov = 1
        output_var = input_var - (input_grad * input_lr + output_accum * input_momentum * input_lr)
    else:
        print("False")
        nesterov = 0
        output_var = input_var - output_accum * input_lr
    #output_data = output_var
    output_data = output_var.astype(np.float32)
    if s_type == 'float16':
        output_accum = output_accum.astype(s_type)
        output_var = output_var.astype(s_type)
        output_data = output_data.astype(s_type)
    return output_var, output_accum


ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
                                                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)})

