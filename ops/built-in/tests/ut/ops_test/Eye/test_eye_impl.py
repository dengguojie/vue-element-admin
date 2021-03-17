# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
import torch

ut_case = BroadcastOpUT("eye")


#pylint: disable=unused-argument
def calc_expect_func(y, num_rows, num_columns, a):
    res = torch.eye(num_rows)
    res = res.numpy()
    print(res)
    return [res, ]


ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "output"}, 3, 3, (16, 16), ],
    "expect": ValueError
})
ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "output"}, 3, 3, (1, 1), 1],
})
ut_case.add_case("all", {
    "params": [{"dtype": "bool", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "output"}, 1, 7, (1, 1)],
    "expect": ValueError
})
ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "output"}, 1, 7, (1, 1)],
})
ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (12, 12), "shape": (12, 12),
                "param_type": "output"}, 65, 63489, (1, 1)],
})
ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (12, 12), "shape": (12, 12),
                "param_type": "output"}, 65, 2, (1, 1)],
})
ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (12, 12), "shape": (12, 12),
                "param_type": "output"}, 7, 2, (4, 4)],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "output"}, 3, 3, (-1, 1)],
    "expect": ValueError
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "output"}, -3, 3, (1, 1)],
    "expect": ValueError
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (253952, 253952), "shape": (253952, 253952),
                "param_type": "output"}, 253952, 253952, (1, 1)],
    "expect": RuntimeError
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (36, 36), "shape": (36, 36),
                "param_type": "output"}, 36, 36, (1, 1)],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "output"}, 3, 3, (1, 1)],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (253, 253), "shape": (253, 253),
                "param_type": "output"}, 253, 253, (1, 1)],
    "calc_expect_func": calc_expect_func
})

