#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("HardShrink", "impl.dynamic.hard_shrink", "hard_shrink")

case1 = {"params": [
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 100)]},
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 100)]},
    0.1
],
    "case_name": "HardShrink_dynamic_fp16_1_unknonwnaxis",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND", "range": [(1, 100)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND", "range": [(1, 100)]},
    0.5
],
    "case_name": "HardShrink_dynamic_fp32_1_2_unknonwnaxis",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case3 = {"params": [
    {"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
    {"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
    1.0
],
    "case_name": "HardShrink_dynamic_fp16_unknowndimension",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case4 = {"params": [
    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
    2.0
],
    "case_name": "HardShrink_dynamic_fp32_unknowndimension",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])