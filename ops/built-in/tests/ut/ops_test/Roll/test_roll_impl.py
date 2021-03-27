# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
import torch
ut_case = BroadcastOpUT("roll")

#pylint: disable=unused-argument
def calc_expect_func(input_x, output_z, shifts, dims):
    res = torch.roll(torch.from_numpy(input_x['value']).float(), shifts, dims).numpy()
    return [res, ]


# [TODO] coding cases here
ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 50), "shape": (3, 4, 50),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 50), "shape": (3, 4, 50),
                "param_type": "output"},
               [20, 50], [0, 1]],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (2, 3, 4), "shape": (2, 3, 4),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (2, 3, 4), "shape": (2, 3, 4),
                "param_type": "output"},
               [20, 50], [0, 1]],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20], []],
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20.0], []],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "int16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "int16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20], [0]],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20.0], [0]],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20], [0.0]],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20.0], [0, 1]],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20], [-3]],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [-20], []],
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [-20, 10], []],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [-20], [-3]],
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20, 1, 5], [0, 1, 2]],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               [20], [-4]],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})
