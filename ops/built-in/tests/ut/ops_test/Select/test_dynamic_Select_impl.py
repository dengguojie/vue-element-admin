#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Select", "impl.dynamic.select", "select")

case1 = {"params": [
    {"shape": (2, ), "dtype": "int8", "format": "ND", "ori_shape": (2, ), "ori_format": "ND", "range": [(2, 2)]},
    {"shape": (1, -1), "dtype": "int32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND", "range": [(1, 1), (1, 100)]},
    {"shape": (1, -1), "dtype": "int32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND", "range": [(1, 1), (1, 100)]},
    {"shape": (1, -1), "dtype": "int32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND", "range": [(1, 1), (1, 100)]},
],
    "case_name": "select_dynamic_1",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")