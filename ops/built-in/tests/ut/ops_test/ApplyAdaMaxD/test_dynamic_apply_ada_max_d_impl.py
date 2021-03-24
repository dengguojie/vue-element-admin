#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
dynamic test
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyAdaMaxD", "impl.dynamic.apply_ada_max_d", "apply_ada_max_d")

case1 = {
    "params": [
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
    ],
    "case_name": "apply_ada_max_d_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
    ],
    "case_name": "apply_ada_max_d_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
    ],
    "case_name": "apply_ada_max_d_3",
    "expect": RuntimeError,
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
