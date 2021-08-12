#!/usr/bin/env/ python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("GlobalLpPool", "impl.dynamic.global_lppool", "global_lppool")

case1 = {
    "params": [
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 4, 5),
         "ori_format": "ND", "range": [(1, 1000), (1, 1000), (1, 1000), (1, 1000)]},
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 1, 1),
         "ori_format": "ND", "range": [(1, 1000), (1, 1000), (1, 1000), (1, 1000)]},
    ],
    "case_name": "global_lppool_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1, -1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 4, 5, 16),
         "ori_format": "ND", "range": [(1, 1000), (1, 1000), (1, 1000), (1, 1000), (1, 1000)]},
        {"shape": (-1, -1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 1, 1, 1),
         "ori_format": "ND", "range": [(1, 1000), (1, 1000), (1, 1000), (1, 1000), (1, 1000)]},
    ],
    "case_name": "global_lppool_dynamic_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case2)

if __name__ == "__main__":
    ut_case.run()
