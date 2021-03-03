# # -*- coding:utf-8 -*-

import numpy as np
from op_test_frame.ut import ElementwiseOpUT
from op_test_frame.common import precision_info

ut_case = ElementwiseOpUT("GridUnnormal", None, None)

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def calc_expect_func(grid, assist, diff, position, align_corners=False):
    grid_value = grid.get("value")
    assist_value = assist.get("value")

    if align_corners:
        tmp1 = (grid_value + 1.0) * (assist_value - 1) * 0.5
    else:
        tmp1 = ((grid_value + 1.0) * assist_value - 1) * 0.5
    pos_t = np.floor(tmp1)
    diff2 = tmp1 - pos_t
    return [diff2, pos_t]


ut_case.add_precision_case("all", {"params": [
    {"shape": (1, 6, 5, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 6, 5, 2), "ori_format": "ND",
     "param_type": "input"},
    {"shape": (1, 6, 5, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 6, 5, 2), "ori_format": "ND",
     "param_type": "input"},
    {"shape": (1, 6, 5, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 6, 5, 2), "ori_format": "ND",
     "param_type": "output"},
    {"shape": (1, 6, 5, 2), "dtype": "int32", "format": "ND", "ori_shape": (1, 6, 5, 2), "ori_format": "ND",
     "param_type": "output"}, ], "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
ut_case.add_precision_case("all", {"params": [
    {"shape": (3, 88, 15, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 88, 15, 2), "ori_format": "ND",
     "param_type": "input"},
    {"shape": (3, 88, 15, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 88, 15, 2), "ori_format": "ND",
     "param_type": "input"},
    {"shape": (3, 88, 15, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 88, 15, 2), "ori_format": "ND",
     "param_type": "output"},
    {"shape": (3, 88, 15, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 88, 15, 2), "ori_format": "ND",
     "param_type": "output"}, ], "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
ut_case.add_precision_case("all", {"params": [
    {"shape": (88, 3, 19, 2), "dtype": "float16", "format": "ND", "ori_shape": (88, 3, 19, 2), "ori_format": "ND",
     "param_type": "input"},
    {"shape": (88, 3, 19, 2), "dtype": "float16", "format": "ND", "ori_shape": (88, 3, 19, 2), "ori_format": "ND",
     "param_type": "input"},
    {"shape": (88, 3, 19, 2), "dtype": "float16", "format": "ND", "ori_shape": (88, 3, 19, 2), "ori_format": "ND",
     "param_type": "output"},
    {"shape": (88, 3, 19, 2), "dtype": "int32", "format": "ND", "ori_shape": (88, 3, 19, 2), "ori_format": "ND",
     "param_type": "output"}, True], "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
