#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
fast_gelu dynamic test UT
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("FastGelu", "impl.dynamic.fast_gelu", "fast_gelu")

case1 = {
    "params": [
        {
            "shape": (-1,),
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
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
    ],
    "case_name": "FastGelu_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {
            "shape": (-1,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
    ],
    "case_name": "FastGelu_2",
    "expect": "success",
    "support_expect": True
}
case3 = {
    "params": [
        {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, 2),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, 2),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100)]
        },
    ],
    "case_name": "FastGelu_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [
        {
            "shape": (-1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 2),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 2),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100)]
        },
    ],
    "case_name": "FastGelu_4",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)
ut_case.add_case("Ascend310", case1)
ut_case.add_case("Ascend310", case2)
ut_case.add_case("Ascend310", case3)
ut_case.add_case("Ascend310", case4)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
    ut_case.run("Ascend310")
