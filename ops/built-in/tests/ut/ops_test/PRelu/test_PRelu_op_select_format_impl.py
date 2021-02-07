#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("PRelu", "impl.prelu", "op_select_format")


def get_input(input_shape, weight_shape, dtype, data_format):
    input_shape = list(input_shape)
    weight_shape = list(weight_shape)
    input_dict = {
        "shape": input_shape,
        "dtype": dtype,
        "format": data_format,
        "ori_shape": input_shape,
        "ori_format": data_format
    }
    weight_dict = {
        "shape": weight_shape,
        "dtype": dtype,
        "format": data_format,
        "ori_shape": weight_shape,
        "ori_format": data_format
    }

    params_list = [input_dict, weight_dict, input_dict]

    return params_list


case1 = {
    "params": get_input((128, 1, 1, 15), (15,), "float16", "NHWC"),
    "case_name": "prelu_op_select_format_4d_2_5hd",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": get_input((1, 128, 128, 15), (1, 128, 1), "float16", "ND"),
    "case_name": "prelu_op_select_format_nd",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}

case3 = {
    "params": get_input((128, 128), (1,), "float16", "NHWC"),
    "case_name": "prelu_op_select_format_nz",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310", "Hi3796CV300ES", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Hi3796CV300ES", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Hi3796CV300ES", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
