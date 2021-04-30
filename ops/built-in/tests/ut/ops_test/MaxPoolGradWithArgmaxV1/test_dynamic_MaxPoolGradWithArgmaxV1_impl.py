#!/usr/bin/env python
# -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("MaxPoolGradWithArgmaxV1", "impl.dynamic.max_pool_grad_with_argmaxv1", "max_pool_grad_with_argmax_v1")


case1 = {"params": [
    {"shape": (1, 4, 112, 560, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 64, 112, 560),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (1, 4, 113, 561, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 64, 113, 561),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (1, 4, 4, 3964, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (1, 64, 4, 3964),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (1, 4, 112, 560, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 64, 112, 560),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1]],
    "case_name": "dynamic_max_pool_grad_with_argmaxv1_00",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1]],
    "case_name": "dynamic_max_pool_grad_with_argmaxv1_01",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case3 = {"params": [
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (35, 2, 1, 1, 16),
     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
    [1, 3, 3, 1], [1, 1, 2, 1], [1, 1, 1, 1]],
    "case_name": "dynamic_max_pool_grad_with_argmaxv1_02",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
