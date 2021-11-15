#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Fills UT test
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("Fills", "impl.dynamic.fills", "fills")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, 1.0],
    "case_name": "Fills_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int8",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, {
        "shape": (-1, -1),
        "dtype": "int8",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, 1.0],
    "case_name": "Fills_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int8",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, {
        "shape": (-1, -1),
        "dtype": "int8",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, None],
    "case_name": "Fills_3",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
