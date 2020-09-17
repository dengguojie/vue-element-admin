#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("FusedMulAddN", None, None)

case1 = {"params": [
    {"shape": (1, 4, 1, 1, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4, 1, 1, 16)},
    {"shape": (1, 4, 1, 1, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4, 1, 1, 16)},
    {"shape": (), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": ()},
    {"shape": (1, 4, 1, 1, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4, 1, 1, 16)}],
    "case_name": "fused_mul_add_n_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {"params": [
    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16)},
    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16)},
    {"shape": (), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": ()},
    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16)}],
    "case_name": "fused_mul_add_n_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run()
