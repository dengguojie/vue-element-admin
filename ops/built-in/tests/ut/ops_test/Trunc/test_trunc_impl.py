# # -*- coding:utf-8 -*-
import numpy
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info

ut_case = BroadcastOpUT("trunc")

#pylint: disable=unused-argument
def calc_expect_func(input_x, output_y):
    res = numpy.trunc(input_x["value"])
    return [res, ]

ut_case.add_case("all", {
    "params": [{
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "input"
    }, {
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "output"
    }],
    "calc_expect_func":calc_expect_func,
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int8",
        "param_type": "input"
    }, {
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int8",
        "param_type": "output"
    }],
    "calc_expect_func":calc_expect_func,
    "case_name": "error1",
    "expect": RuntimeError
})

ut_case.add_precision_case("all", {
    "params": [{
        "shape": (32, ),
        "ori_shape": (32, ),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "input",
        "value_range":[1,50]
    }, {
        "shape": (32, ),
        "ori_shape": (32, ),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "output",
        "value_range":[1,50]
    }],
    "calc_expect_func": calc_expect_func
})
