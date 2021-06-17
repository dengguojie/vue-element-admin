#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Log1p", "impl.dynamic.log1p", "log1p")

case1 = {"params": [
    {"shape": (-1, -1, 2), "ori_shape": (2, 10, 5), "range": ((1, None), (1, None), (2, 2)), "format": "NHWC",
     "ori_format": "NHWC", "dtype": "float32"},
    {"shape": (-1, -1, 2), "ori_shape": (2, 10, 5), "range": ((1, None), (1, None), (2, 2)), "format": "NHWC",
     "ori_format": "NHWC", "dtype": "float32"}],
    "case_name": "log1p_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {"params": [
    {"shape": (-1,), "ori_shape": (100,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float32"},
    {"shape": (-1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float32"}],
    "case_name": "log1p_dynamic_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case3 = {"params": [
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     "dtype": "float16"},
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     "dtype": "float16"}],
    "case_name": "log1p_dynamic_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case4 = {"params": [
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     "dtype": "float16"},
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     "dtype": "float16"}],
    "case_name": "log1p_dynamic_4",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case5 = {"params": [
    {"shape": (-1, 1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float16"},
    {"shape": (-1, 1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float16"}],
    "case_name": "log1p_dynamic_5",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case6 = {"params": [
    {"shape": (-1, 1), "ori_shape": (2, 2), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float16"},
    {"shape": (-1, 1), "ori_shape": (2, 2), "range": ((2, 2), (2, 2)), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float16"}],
    "case_name": "log1p_dynamic_6",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case7 = {"params": [
    {"shape": (-1, 1), "ori_shape": (2, 2), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float32"},
    {"shape": (-1, 1), "ori_shape": (2, 2), "range": ((2, 2), (2, 2)), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float32"}],
    "case_name": "log1p_dynamic_7",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case8 = {"params": [
    {"shape": (-1, 1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float32"},
    {"shape": (-1, 1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     "dtype": "float32"}],
    "case_name": "log1p_dynamic_8",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910"])
