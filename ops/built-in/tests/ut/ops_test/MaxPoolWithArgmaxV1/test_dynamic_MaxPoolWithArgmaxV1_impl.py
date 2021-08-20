#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
ut of dynamic maxpoolwithargmaxv1
"""

from op_test_frame.ut import OpUT

ut_case = OpUT("MaxPoolWithArgmaxV1", "impl.dynamic.max_pool_with_argmaxv1", "max_pool_with_argmax_v1")

case1 = {"params": [
    {"shape": (32, 1, 256, 256, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 16, 256, 256),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (32, 1, 128, 128, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 16, 128, 128),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (32, 1, 128, 128, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 16, 128, 128),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1]],
    "case_name": "dynamic_max_pool_with_argmaxv1_00",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1]],
    "case_name": "dynamic_max_pool_with_argmaxv1_01",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case3 = {"params": [
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1]],
    "case_name": "dynamic_max_pool_with_argmaxv1_02",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
