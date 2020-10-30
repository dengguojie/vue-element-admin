#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("LogicalOr", "impl.dynamic.logical_or", "logical_or")

case1 = {"params": [
    {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int8"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int8"},
    {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "int8"}],
         "case_name": "logical_or_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "int8"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int8"},
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "int8"}],
         "case_name": "logical_or_dynamic_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "int8"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int8"},
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "int8"}],
         "case_name": "logical_or_dynamic_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "ND", "ori_format": "ND",
     'dtype': "int8"},
    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "ND", "ori_format": "ND", 'dtype': "int8"},
    {"shape": (-1, 10), "ori_shape": (2, 10), "range": ((1, None), (10, 10)), "format": "ND", "ori_format": "ND",
     'dtype': "int8"}],
         "case_name": "logical_or_dynamic_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
