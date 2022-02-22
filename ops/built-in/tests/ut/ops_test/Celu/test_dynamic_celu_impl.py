#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("Celu", "impl.dynamic.celu", "celu")

ut_case.add_case("all", {"params": [
    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (5, 13), 
    "ori_format": "ND", "range": [(1, None), (1, None)]},
    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (5, 13), 
    "ori_format": "ND", "range": [(1, None), (1, None)]}],
    "case_name": "test_dynamic_celu_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True})


if __name__ == "__main__":
    ut_case.run("Ascend910A")
