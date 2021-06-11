# # -*- coding:utf-8 -*-
import torch
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("cross")


# pylint: disable=unused-argument
def calc_expect_func(x1, x2, y, dim):
    x1_tensor = torch.from_numpy(x1["value"])
    x2_tensor = torch.from_numpy(x2["value"])
    res_tensor = torch.cross(x1_tensor, x2_tensor)
    res = res_tensor.numpy()
    return [res, ]


ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 3),
                "shape": (4, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 3),
                "shape": (4, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 3),
                "shape": (4, 3),
                "param_type": "output"}, 1],
    "case_name": "test_cross_precision_case_1",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 3, 3),
                "shape": (2, 3, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 3, 3),
                "shape": (2, 3, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 3, 3),
                "shape": (2, 3, 3),
                "param_type": "output"}, 1],
    "case_name": "test_cross_precision_case_2",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 5, 3),
                "shape": (4, 5, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 5, 3),
                "shape": (4, 5, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 5, 3),
                "shape": (4, 5, 3),
                "param_type": "output"}, 2],
    "case_name": "test_cross_precision_case_3",
    "calc_expect_func": calc_expect_func
})