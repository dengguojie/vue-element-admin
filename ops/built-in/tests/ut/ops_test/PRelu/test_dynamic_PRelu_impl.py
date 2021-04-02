#!/usr/bin/env/ python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

# pylint: disable=invalid-name
ut_case = OpUT("PRelu", "impl.dynamic.prelu", "prelu")


case1 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]},
    {"shape": (-1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,),
     "ori_format": "NHWC", "range": [(1, 1)]},
    {"shape": (1, -1, -1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]}],
         "case_name": "prelu_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]},
    {"shape": (-1, -1, -1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 4, 4),
     "ori_format": "NHWC", "range": [(1, 1), (1, 10), (1, 10)]},
    {"shape": (1, -1, -1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]}],
          "case_name": "prelu_dynamic_2",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}

case3 = {"params": [
    {"shape": (-1, 2, 6, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (64, 2, 4, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
    {"shape": (-1, 2, 6, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1, 2, 1, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
    {"shape": (-1, 2, 6, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (64, 2, 4, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]}],
         "case_name": "prelu_dynamic_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [
    {"shape": (-1, 2, -1, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (64, 2, 4, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]},
    {"shape": (-1,), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,),
     "ori_format": "NDC1HWC0", "range": [(1, 1)]},
    {"shape": (-1, 2, -1, 4, 2, -1), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (64, 2, 4, 4, 2, 16),
     "ori_format": "NDC1HWC0", "range": [(1, 100), (2, 2), (4, 4), (4, 4), (2, 2), (1, 16)]}],
         "case_name": "prelu_dynamic_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
