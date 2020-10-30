# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
import numpy as np

ut_case = BroadcastOpUT("stn_compute")

# [TODO] coding expect function here


# [TODO] coding cases here


ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 5, 5, 16),
                "shape": (1, 1, 5, 5, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 5 * 5 * 4),
                "shape": (1, 1, 5 * 5 * 4), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 5 * 5 * 4),
                "shape": (1, 1, 5 * 5 * 4), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 5, 5, 16),
                "shape": (1, 1, 5, 5, 16),
                "param_type": "output"},
               (5, 5, 5, 5), False],
    "expect": True,
    "case_name": "StnCompute1",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 2, 7, 5, 16),
                "shape": (1, 2, 7, 5, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 2, 7 * 5 * 4),
                "shape": (1, 2, 7 * 5 * 4), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 2, 7 * 5 * 4),
                "shape": (1, 2, 7 * 5 * 4), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 2, 7, 5, 16),
                "shape": (1, 2, 7, 5, 16),
                "param_type": "output"},
               (7, 5, 7, 5), False],
    "expect": True,
    "case_name": "StnCompute2",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 2, 227, 227, 16),
                "shape": (1, 2, 227, 227, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 2, 227 * 227 * 4),
                "shape": (1, 2, 227 * 227 * 4), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 2, 227 * 227 * 4),
                "shape": (1, 2, 227 * 227 * 4), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 2, 227, 227, 16),
                "shape": (1, 2, 227, 227, 16),
                "param_type": "output"},
               (227, 227, 227, 227), False],
    "expect": True,
    "case_name": "StnCompute3",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 2, 500, 500, 16),
                "shape": (1, 2, 500, 500, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 2, 500 * 500 * 4),
                "shape": (1, 2, 500 * 500 * 4), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 2, 500 * 500 * 4),
                "shape": (1, 2, 500 * 500 * 4), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 2, 500, 500, 16),
                "shape": (1, 2, 500, 500, 16),
                "param_type": "output"},
               (500, 500, 500, 500), False],
    "expect": True,
    "case_name": "StnCompute4",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 2, 2, 16),
                "shape": (1, 1, 2, 2, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2 * 2 * 4),
                "shape": (1, 1, 2 * 2 * 4), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2 * 2 * 4),
                "shape": (1, 1, 2 * 2 * 4), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 1, 2, 2, 16),
                "shape": (1, 1, 2, 2, 16),
                "param_type": "output"},
               (2, 2, 2, 2), False],
    "expect": True,
    "case_name": "StnCompute5",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 17, 2, 2, 16),
                "shape": (1, 17, 2, 2, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 17, 2 * 2 * 4),
                "shape": (1, 17, 2 * 2 * 4), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 17, 2 * 2 * 4),
                "shape": (1, 17, 2 * 2 * 4), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 17, 2, 2, 16),
                "shape": (1, 17, 2, 2, 16),
                "param_type": "output"},
               (2, 2, 2, 2), False],
    "expect": True,
    "case_name": "StnCompute7",
    "format_expect": [],
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 10000, 2, 2, 16),
                "shape": (1, 10000, 2, 2, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 10000, 2 * 2 * 4),
                "shape": (1, 10000, 2 * 2 * 4), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 10000, 2 * 2 * 4),
                "shape": (1, 10000, 2 * 2 * 4), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1, 10000, 2, 2, 16),
                "shape": (1, 10000, 2, 2, 16),
                "param_type": "output"},
               (2, 2, 2, 2), False],
    "expect": True,
    "case_name": "StnCompute6",
    "format_expect": [],
})

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Hi3796CV300CS"])
    exit(0)
