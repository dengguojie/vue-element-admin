#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TileWithAxis", "impl.dynamic.tile_with_axis", "tile_with_axis")

case1 = {
    "params": [
        {
            "shape": (-1, 2, 4), 
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (1, 2, 4),
            "ori_format": "NCHW",
            "range": [(1, None), (1, None), (1, None)]
        },
        {
            "shape": (-1, 2, 4),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (2, 2, 4),
            "ori_format": "NCHW",
            "range": [(1, None), (1, None), (1, None)]
        },
        2,
        1
    ],
    "case_name": "TileWithAxis_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {
            "shape": (-1, 2, 4), 
            "dtype": "float32",
            "format": "NCHW",
            "ori_shape": (1, 2, 4),
            "ori_format": "NCHW",
            "range": [(1, None), (1, None), (1, None)]
        },
        {
            "shape": (-1, 2, 4),
            "dtype": "float32",
            "format": "NCHW",
            "ori_shape": (2, 2, 4),
            "ori_format": "NCHW",
            "range": [(1, None), (1, None), (1, None)]
        },
        1,
        1
    ],
    "case_name": "TileWithAxis_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {
            "shape": (-1, 1, 13, 4, 16), 
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (1, 16, 13, 4),
            "ori_format": "NCHW",
            "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]
        },
        {
            "shape": (-1, 1, 13, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (2, 16, 13, 4),
            "ori_format": "NCHW",
            "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]
        },
        2,
        0
    ],
    "case_name": "TileWithAxis_3",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
