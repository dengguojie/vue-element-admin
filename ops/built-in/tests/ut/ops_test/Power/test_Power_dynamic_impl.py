#!/usr/bin/env/ python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("Power", "impl.dynamic.power", "power")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1,100)]},
    ],
    "case_name": "power_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1,100)]},
        1.0,
        3.0
    ],
    "case_name": "power_dynamic_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case3 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1,100)]},
        0.0,
        0.0
    ],
    "case_name": "power_dynamic_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case4 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1,100)]},
        4.0,
        1.0
    ],
    "case_name": "power_dynamic_4",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case3)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case4)

if __name__ == "__main__":
    ut_case.run()
