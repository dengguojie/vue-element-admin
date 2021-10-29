# # -*- coding:utf-8 -*-
import sys
import numpy as np
from op_test_frame.ut import ElementwiseOpUT

ut_case = ElementwiseOpUT("expand_d")

def calc_expect_func(x, y, shape):
    print(x.get("value"))
    print(shape)
    shape_x = list(x.get("shape"))
    print(shape_x)
    shape = list(shape)
    if len(shape_x) < len(shape):
        tmp = shape_x.copy()
        shape_x = shape.copy()
        shape = tmp.copy()
    target_shape = []
    if len(shape_x) != len(shape):
        diff = len(shape_x) - len(shape)
        shape = [1] * diff + shape
    for i in range(len(shape_x)):
        if shape_x[i] > shape[i]:
            target_shape.append(shape_x[i])
        else:
            target_shape.append(shape[i])
    print(target_shape)
    res = np.broadcast_to(x.get("value"), target_shape)
    print(res)
    return res


ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 3), "shape": (1, 3),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 3), "shape": (3, 2, 3),
                "param_type": "output"},
               [3, 2, 3]],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_expand_d_01"
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 3), "shape": (3, 2, 3),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 3), "shape": (3, 2, 3),
                "param_type": "output"},
               [3, 2, 3]],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_expand_d_02"
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 3), "shape": (3, 2, 3),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 3), "shape": (3, 2, 3),
                "param_type": "output"},
               [3, 2, 3]],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_expand_d_03"
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 3), "shape": (3, 2, 3),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 3), "shape": (3, 2, 3),
                "param_type": "output"},
               [3, 2, 3]],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_expand_d_04"
})
