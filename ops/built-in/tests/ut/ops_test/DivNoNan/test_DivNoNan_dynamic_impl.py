#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("DivNoNan", "impl.dynamic.div_no_nan", "div_no_nan")

case1 = {"params": [{"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float32"},
                    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float32"},
                    {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float32"}],
         "case_name": "div_no_nan_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float32"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"},
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float32"}],
    "case_name": "div_no_nan_dynamic_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case3 = {"params": [
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float16"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"},
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float16"}],
    "case_name": "div_no_nan_dynamic_3",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True}

case4 = {"params": [
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "ND", "ori_format": "ND",
     'dtype': "float16"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "ND", "ori_format": "ND", 'dtype': "float16"},
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "ND", "ori_format": "ND",
     'dtype': "float16"}],
    "case_name": "div_no_nan_dynamic_4",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case5 = {"params": [
    {"shape": (-1, 10, -1), "ori_shape": (2, 10, 3), "range": ((1, None), (10, 10), (1, None)), "format": "ND",
     "ori_format": "ND", 'dtype': "float16"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "ND", "ori_format": "ND", 'dtype': "float16"},
    {"shape": (-1, 10, -1), "ori_shape": (2, 10, 3), "range": ((1, None), (10, 10), (1, None)), "format": "ND",
     "ori_format": "ND", 'dtype': "float16"}],
    "case_name": "div_no_nan_dynamic_5",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case6 = {"params": [
    {"shape": (-1, 10, -1), "ori_shape": (2, 10, 3), "range": ((1, None), (10, 10), (1, None)), "format": "ND",
     "ori_format": "ND", 'dtype': "float32"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "ND", "ori_format": "ND", 'dtype': "float32"},
    {"shape": (-1, 10, -1), "ori_shape": (2, 10, 3), "range": ((1, None), (10, 10), (1, None)), "format": "ND",
     "ori_format": "ND", 'dtype': "float32"}],
    "case_name": "div_no_nan_dynamic_6",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case7 = {"params": [
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", 'dtype': "float32"},
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", 'dtype': "float32"},
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", 'dtype': "float32"}],
    "case_name": "div_no_nan_dynamic_7",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
