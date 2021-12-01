# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("MishGrad", None, None)

shape_1 = (10, 1023)
data_type = "float16"
data_format = "NCHW"
case1 = {
    "params": [
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format}
    ],
    "case_name": "mish_grad_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

data_type = "float32"
case2 = {
    "params": [
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        None,
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format}
    ],
    "case_name": "mish_grad_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}


case3 = {
    "params": [
        {"shape": shape_1, "dtype": "int32", "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format}
    ],
    "case_name": "mish_grad_3",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}

case4 = {
    "params": [
        {"shape": (1, ), "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        None,
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format}
    ],
    "case_name": "mish_grad_4",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}

case5 = {
    "params": [
        {"shape": shape_1, "dtype": data_type, "format": "ND", "ori_shape": shape_1, "ori_format": data_format},
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        None,
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format}
    ],
    "case_name": "mish_grad_5",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend920A"], case1)
ut_case.add_case(["Ascend910A", "Ascend920A"], case2)
ut_case.add_case(["Ascend910A", "Ascend920A"], case3)
ut_case.add_case(["Ascend910A", "Ascend920A"], case4)
ut_case.add_case(["Ascend910A", "Ascend920A"], case5)


def calc_expect_func(input_grad, input_x, input_tanhx):
    if input_tanhx is None:
        exp_x = np.exp(input_x)
        exp_add_x = exp_x + 1
        rec_exp_add_x = 1.0 / (exp_add_x * exp_add_x + 1.0)
        result_1 = exp_add_x * exp_x * input_x * rec_exp_add_x * rec_exp_add_x * 4
        result = result_1 - 2 * rec_exp_add_x + 1
    else:
        pow_input_tanhx = input_tanhx * input_tanhx
        result = input_tanhx + input_x * (1 - pow_input_tanhx) * np.exp(input_x) / (1 + np.exp(input_x))
    result = result * input_grad
    return [result, ]


def compute_tanhx(data):
    data_1 = 1 + np.exp(data)
    data_2 = data_1 * data_1
    data_result = (data_2 - 1) / (data_2 + 1)
    return data_result


if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend920A"])
    # ut_case.run()
    exit(0)
