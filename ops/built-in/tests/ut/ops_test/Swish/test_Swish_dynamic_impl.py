#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Swish", "impl.dynamic.swish", "swish")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
         "ori_format": "ND", "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
         "ori_format": "ND", "range": [(1, 100)]},
    ],
    "case_name": "Swish_1",
    "expect": "success"
}

case2 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
         "ori_format": "ND", "range": [(1, 100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
         "ori_format": "ND", "range": [(1, 100)]}
    ],
    "case_name": "Swish_2",
    "expect": "success"
}
case3 = {
    "params": [
        {"shape": (-1, -1), "dtype": "float32", "format": "ND",
         "ori_shape": (2, 2), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
        {"shape": (-1, -1), "dtype": "float32", "format": "ND",
         "ori_shape": (2, 2), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
        -3.0],
    "case_name": "Swish_3",
    "expect": "success"
}

case4 = {
    "params": [
        {"shape": (-1, -1), "dtype": "float32", "format": "ND",
         "ori_shape": (2, 2), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
        {"shape": (-1, -1), "dtype": "float32", "format": "ND",
         "ori_shape": (2, 2), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
    ],
    "case_name": "Swish_4",
    "expect": "success"
}
ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend310"], case3)
ut_case.add_case(["Ascend910A", "Ascend310"], case4)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
