#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import itertools

import numpy as np

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from tbe.dsl.compute.dilation_compute import shape_align

ut_case = OpUT("dilation")


def calc_expect_func(input_x, output_z, dilations, pads, padding_value):
    shape_x = input_x.get("shape")
    value_x = input_x.get("value")
    shape_aligned_x = shape_align(shape_x, input_x.get("dtype"))
    shape_dilated = list(map(lambda a, b: a * b, shape_aligned_x, dilations))
    shape_cut = list(map(lambda a, b: a - b + 1, shape_dilated, dilations))
    shape_cut[-1] = shape_x[-1]
    res = np.full(shape_cut, padding_value, dtype=input_x.get("dtype"))
    value_x_aligned = np.full(shape_aligned_x, 0, dtype=input_x.get("dtype"))
    indices_x = itertools.product(*list(list(range(size)) for size in shape_x))
    for index in indices_x:
        value = value_x.__getitem__(index)
        value_x_aligned.__setitem__(index, value)

    indices = itertools.product(*list(list(range(size)) for size in shape_cut))
    for index in indices:
        if all(list((index[i] % dilations[i] == 0) for i in range(len(index)))):
            value = value_x_aligned.__getitem__(tuple(index[i] // dilations[i] for i in range(len(index))))
            res.__setitem__(index, value)
    return res


precision_case_list = [
    ['float16', [8, 10, 3], [3, 5, 1], 0.24487, [22, 46, 3]],
    ['float32', [2, 2, 9, 13, 5], [1, 1, 2, 4, 1], 1.19492, [2, 2, 17, 49, 5]],
    ['float16', [1, 3, 9, 4, 1, 6], [1, 1, 4, 3, 1, 1], 2.52139, [1, 3, 33, 10, 1, 6]],
    ['int8', [7, 3], [1, 1], 2.0096, [7, 3]],
    ['float16', [1, 3, 3], [4, 4, 1], 4.77789, [1, 9, 3]],
    ['float32', [9, 10, 1], [1, 2, 1], 4.14656, [9, 19, 1]],
    ['float32', [5, 1, 6, 6], [2, 5, 2, 1], 3.34796, [9, 1, 11, 6]],
    ['float16', [10, 7, 5], [5, 5, 1], 4.51234, [46, 31, 5]],
    ['int8', [8, 3, 1], [5, 3, 1], 4.11004, [36, 7, 1]],
    ['float32', [10, 6], [2, 1], 4.1219, [19, 6]],
    ['int8', [3, 1, 8, 1, 1], [1, 1, 1, 5, 1], 2.57715, [3, 1, 8, 1, 1]],
    ['int8', [6, 5, 1, 2], [4, 3, 2, 1], 4.23498, [21, 13, 1, 2]],
    ['int8', [8, 1, 8, 4], [4, 3, 2, 1], 0.39946, [29, 1, 15, 4]],
    ['float16', [8, 3, 9, 4], [3, 4, 2, 1], 4.84731, [22, 9, 17, 4]],
    ['int8', [2, 2, 24, 34, 9], [1, 1, 3, 1, 1], 3.68147, [2, 2, 70, 34, 9]],
    ['float16', [3, 3, 3, 32, 4], [1, 1, 1, 4, 1], 3.69438, [3, 3, 3, 125, 4]],
    ['float16', [8, 5], [4, 1], 4.75678, [29, 5]],
    ['int8', [7, 7, 3, 1], [1, 4, 1, 1], 0, [7, 25, 3, 1]]
]

runtime_error_case_list = [
    ['float16', [8, 10, 3], [3, 5], 0.24487, [22, 46, 3]],
    ['float32', [2, 2, 9, 13], [1, 1, 2, 4, 1], 1.19492, [2, 2, 17, 49, 5]],
    ['float32', [2, 2, 9, 13, 5], [1, 1, 2, 4, 0], 1.19492, [2, 2, 17, 49, 5]],
    ['float32', [2, 2, 9, 13, 5], [1, 1, 2, -1, 1], 1.19492, [2, 2, 17, 49, 5]],
    ['int8', [2, 2, 24, 340, 9], [1, 1, 3, 1, 1], 3.68147, [2, 2, 70, 34, 9]],
    ['int8', None, [1, 1, 3, 1, 1], 3.68147, [2, 2, 70, 34, 9]],
    ['None', [2, 2, 24, 34, 9], [1, 1, 3, 1, 1], 3.68147, [2, 2, 70, 34, 9]],
    ['float16', [8, 3, 9, 4], [3, 4.5, 2, 1], 4.84731, [22, 9, 17, 4]],
    ['int32', [6, 5, 1, 2], [4, 3, 2, 1], 4.23498, [21, 13, 1, 2]],
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
                "shape": case[4],
                "param_type": "output"
            },
            case[2],
            None,
            case[3],
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
                "shape": case[4],
                "param_type": "output"
            },
            case[2],
            None,
            case[3],
        ],
        "expect": RuntimeError
    }
    ut_case.add_case("all", case_info)

if __name__ == "__main__":
    ut_case.run()
