#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import itertools

import numpy as np

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from impl.dilation import get_op_support_info

ut_case = OpUT("Dilation", "impl.dilation", "dilation")

def calc_expect_func(input_x, output_z, dilations, pads, padding_value):
    shape_x = input_x.get("shape")
    value_x = input_x.get("value")
    if pads is None:
        pads = [0, 0, 0, 0]
    shape_res = list(map(lambda a, b: (a - 1)*b + 1, shape_x, dilations))
    shape_res[2] += (pads[0] + pads[1])
    shape_res[3] += (pads[2] + pads[3])
    res = np.full(shape_res, padding_value, dtype=input_x.get("dtype"))

    for n in range(shape_res[0]):
        for c1 in range(shape_res[1]):
            for h in range(shape_res[2]):
                for w in range(shape_res[3]):
                    for c0 in range(shape_res[4]):
                        if (h - pads[0]) % dilations[2] == 0 and (w - pads[2]) % dilations[3] == 0:
                            res[n, c1, h, w, c0] = value_x[n, c1,
                                                          (h - pads[0]) // dilations[2],
                                                          (w - pads[2]) // dilations[3],
                                                          c0]
    return res


precision_case_list = [
    #['float16', [1, 2, 7, 7, 16], [1, 1, 2, 2, 1],  None, 0, [1, 2, 13, 13, 16]],
    ['float16', [1, 2, 7, 7, 16], [1, 1, 2, 2, 1], [0, 1, 0, 1], 0, [1, 2, 14, 14, 16]],
    #['float16', [1, 2, 7, 7, 16], [1, 1, 2, 2, 1], [], 0, [1, 2, 13, 13, 16]]
]

runtime_error_case_list = [
    ['float16', [1, 2, 7, 7, 16], [1, 2, 2, 1], None, 0, [1, 2, 13, 13, 16]],
    ['float64', [1, 2, 7, 7, 16], [1, 1, 2, 2, 1], None, 0, [1, 2, 14, 14, 16]],
    ['float16', [1, 2, 7, 7, 16], [1, 1, 2, 2, 1], [0, 1, 0, 1, 0], 0, [1, 2, 14, 14, 16]],
]

for case in precision_case_list:
    case_info = {
        "params": [
            {
                "dtype": case[0],
                "format": "ND",
                "shape": case[1],
                "param_type": "input"
            },
            {
                "dtype": case[0],
                "format": "ND",
                "shape": case[5],
                "param_type": "output"
            },
            case[2],
            case[3],
            case[4],
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    }
    ut_case.add_precision_case("all", case_info)

for case in runtime_error_case_list:
    case_info = {
        "params": [
            {
                "dtype": case[0],
                "format": "ND",
                "shape": case[1],
                "param_type": "input"
            },
            {
                "dtype": case[0],
                "format": "ND",
                "shape": case[5],
                "param_type": "output"
            },
            case[2],
            case[3],
            case[4],
        ],
        "expect": RuntimeError
    }
    ut_case.add_case("all", case_info)

def test_dilation_support_info(test_arg):
    x = {"shape": (4, 1, 3, 3, 16), "ori_shape": (4, 16 , 3, 3), "dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC"}
    y = {"shape": (4, 1, 5, 5, 16), "ori_shape": (4, 16 , 5, 5), "dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC"}
    dilations = (1, 1, 2, 2, 1)
    get_op_support_info(x, y, dilations)
ut_case.add_cust_test_func(test_func=test_dilation_support_info)


if __name__ == "__main__":
    ut_case.run()
