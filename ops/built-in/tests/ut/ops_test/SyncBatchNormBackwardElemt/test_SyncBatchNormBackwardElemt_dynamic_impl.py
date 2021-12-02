#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic SyncBatchNormBackwardElemt ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("SyncBatchNormBackwardElemt", "impl.dynamic.sync_batch_norm_backward_elemt",
               "sync_batch_norm_backward_elemt")

def calc_expect_func(grad_output, save_input, mean, invstd, weight, mean_dy, mean_dy_xmu, grad_input):
    """
    calc_expect_func
    """
    grad_output = grad_output.get("value")
    save_input = save_input.get("value")
    mean = mean.get("value")
    invstd = invstd.get("value")
    weight = weight.get("value")
    mean_dy = mean_dy.get("value")
    mean_dy_xmu = mean_dy_xmu.get("value")

    output_dy = grad_output - mean_dy
    input_mean = save_input - mean
    invstd_sq = invstd * invstd
    invstd_dy_xmu = invstd_sq * mean_dy_xmu
    input_invstd = input_mean * invstd_dy_xmu
    ouput_input = output_dy - input_invstd
    invstd_w = invstd * weight
    grad_input = ouput_input * invstd_w
    return grad_input

case1 = {"params": [
         {"shape": (-1, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 2), "ori_format": "ND",
          "range": [(3, 3), (2, 2)]},
         {"shape": (-1, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 2), "ori_format": "ND",
          "range": [(3, 3), (2, 2)]},
         {"shape": (-1, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 2), "ori_format": "ND",
          "range": [(3, 3), (2, 2)]},
         {"shape": (-1, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 2), "ori_format": "ND",
          "range": [(3, 3), (2, 2)]},
         {"shape": (-1, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 2), "ori_format": "ND",
          "range": [(3, 3), (2, 2)]},
         {"shape": (-1, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 2), "ori_format": "ND",
          "range": [(3, 3), (2, 2)]},
         {"shape": (-1, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 2), "ori_format": "ND",
          "range": [(3, 3), (2, 2)]},
         {"shape": (-1, 2), "dtype": "float16", "format": "ND", "ori_shape": (3, 2), "ori_format": "ND",
          "range": [(3, 3), (2, 2)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_1",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
         "expect": "success",
         "support_expect": True}

case2 = {"params": [
         {"shape": (2, -1, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4)]},
         {"shape": (2, -1, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4)]},
         {"shape": (2, -1, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4)]},
         {"shape": (2, -1, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4)]},
         {"shape": (2, -1, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4)]},
         {"shape": (2, -1, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4)]},
         {"shape": (2, -1, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4)]},
         {"shape": (2, -1, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_2",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
         "expect": "success",
         "support_expect": True}

case3 = {"params": [
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_3",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
         "expect": "success",
         "support_expect": True}

case4 = {"params": [
         {"shape": (1, 2, 4, 3, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 3, 3), "ori_format": "ND",
          "range": [(1, 1), (2, 2), (4, 4), (3, 3), (3, 3)]},
         {"shape": (1, 2, 4, 3, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 3, 3), "ori_format": "ND",
          "range": [(1, 1), (2, 2), (4, 4), (3, 3), (3, 3)]},
         {"shape": (1, 2, 4, 3, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 3, 3), "ori_format": "ND",
          "range": [(1, 1), (2, 2), (4, 4), (3, 3), (3, 3)]},
         {"shape": (1, 2, 4, 3, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 3, 3), "ori_format": "ND",
          "range": [(1, 1), (2, 2), (4, 4), (3, 3), (3, 3)]},
         {"shape": (1, 2, 4, 3, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 3, 3), "ori_format": "ND",
          "range": [(1, 1), (2, 2), (4, 4), (3, 3), (3, 3)]},
         {"shape": (1, 2, 4, 3, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 3, 3), "ori_format": "ND",
          "range": [(1, 1), (2, 2), (4, 4), (3, 3), (3, 3)]},
         {"shape": (1, 2, 4, 3, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 3, 3), "ori_format": "ND",
          "range": [(1, 1), (2, 2), (4, 4), (3, 3), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_4",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
         "expect": "success",
         "support_expect": True}

case5 = {"params": [
         {"shape": (-1, 3, 5, 11, 13, 17), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 5, 11, 13, 17), "ori_format": "ND",
          "range": [(1, 1), (3, 3), (5, 5), (11, 11), (13, 13), (17, 17)]},
         {"shape": (-1, 3, 5, 11, 13, 17), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 5, 11, 13, 17), "ori_format": "ND",
          "range": [(1, 1), (3, 3), (5, 5), (11, 11), (13, 13), (17, 17)]},
         {"shape": (-1, 3, 5, 11, 13, 17), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 5, 11, 13, 17), "ori_format": "ND",
          "range": [(1, 1), (3, 3), (5, 5), (11, 11), (13, 13), (17, 17)]},
         {"shape": (-1, 3, 5, 11, 13, 17), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 5, 11, 13, 17), "ori_format": "ND",
          "range": [(1, 1), (3, 3), (5, 5), (11, 11), (13, 13), (17, 17)]},
         {"shape": (-1, 3, 5, 11, 13, 17), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 5, 11, 13, 17), "ori_format": "ND",
          "range": [(1, 1), (3, 3), (5, 5), (11, 11), (13, 13), (17, 17)]},
         {"shape": (-1, 3, 5, 11, 13, 17), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 5, 11, 13, 17), "ori_format": "ND",
          "range": [(1, 1), (3, 3), (5, 5), (11, 11), (13, 13), (17, 17)]},
         {"shape": (-1, 3, 5, 11, 13, 17), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 5, 11, 13, 17), "ori_format": "ND",
          "range": [(1, 1), (3, 3), (5, 5), (11, 11), (13, 13), (17, 17)]},
         {"shape": (-1, 3, 5, 11, 13, 17), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 5, 11, 13, 17), "ori_format": "ND",
          "range": [(1, 1), (3, 3), (5, 5), (11, 11), (13, 13), (17, 17)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_5",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
         "expect": "success",
         "support_expect": True}

case6 = {"params": [
         {"shape": (-1, 8, 8, 16, 8, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 8, 16, 8, 16, 16), "ori_format": "ND",
          "range": [(8, 8), (8, 8), (8, 8), (16, 16), (8, 8), (16, 16), (16, 16)]},
         {"shape": (-1, 8, 8, 16, 8, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 8, 16, 8, 16, 16), "ori_format": "ND",
          "range": [(8, 8), (8, 8), (8, 8), (16, 16), (8, 8), (16, 16), (16, 16)]},
         {"shape": (-1, 8, 8, 16, 8, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 8, 16, 8, 16, 16), "ori_format": "ND",
          "range": [(8, 8), (8, 8), (8, 8), (16, 16), (8, 8), (16, 16), (16, 16)]},
         {"shape": (-1, 8, 8, 16, 8, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 8, 16, 8, 16, 16), "ori_format": "ND",
          "range": [(8, 8), (8, 8), (8, 8), (16, 16), (8, 8), (16, 16), (16, 16)]},
         {"shape": (-1, 8, 8, 16, 8, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 8, 16, 8, 16, 16), "ori_format": "ND",
          "range": [(8, 8), (8, 8), (8, 8), (16, 16), (8, 8), (16, 16), (16, 16)]},
         {"shape": (-1, 8, 8, 16, 8, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 8, 16, 8, 16, 16), "ori_format": "ND",
          "range": [(8, 8), (8, 8), (8, 8), (16, 16), (8, 8), (16, 16), (16, 16)]},
         {"shape": (-1, 8, 8, 16, 8, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 8, 16, 8, 16, 16), "ori_format": "ND",
          "range": [(8, 8), (8, 8), (8, 8), (16, 16), (8, 8), (16, 16), (16, 16)]},
         {"shape": (-1, 8, 8, 16, 8, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 8, 16, 8, 16, 16), "ori_format": "ND",
          "range": [(8, 8), (8, 8), (8, 8), (16, 16), (8, 8), (16, 16), (16, 16)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_6",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
         "expect": "success",
         "support_expect": True}

case7 = {"params": [
         {"shape": (-1, 1888, 71, 57), "dtype": "float16", "format": "ND", "ori_shape": (16, 1888, 71, 57), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (71, 71), (57, 57)]},
         {"shape": (-1, 1888, 71, 57), "dtype": "float16", "format": "ND", "ori_shape": (16, 1888, 71, 57), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (71, 71), (57, 57)]},
         {"shape": (-1, 1888, 71, 57), "dtype": "float16", "format": "ND", "ori_shape": (16, 1888, 71, 57), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (71, 71), (57, 57)]},
         {"shape": (-1, 1888, 71, 57), "dtype": "float16", "format": "ND", "ori_shape": (16, 1888, 71, 57), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (71, 71), (57, 57)]},
         {"shape": (-1, 1888, 71, 57), "dtype": "float16", "format": "ND", "ori_shape": (16, 1888, 71, 57), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (71, 71), (57, 57)]},
         {"shape": (-1, 1888, 71, 57), "dtype": "float16", "format": "ND", "ori_shape": (16, 1888, 71, 57), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (71, 71), (57, 57)]},
         {"shape": (-1, 1888, 71, 57), "dtype": "float16", "format": "ND", "ori_shape": (16, 1888, 71, 57), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (71, 71), (57, 57)]},
         {"shape": (-1, 1888, 71, 57), "dtype": "float16", "format": "ND", "ori_shape": (16, 1888, 71, 57), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (71, 71), (57, 57)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_7",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
         "expect": "success",
         "support_expect": True}

case8 = {"params": [
         {"shape": (-1, 1888, 11, 3), "dtype": "float32", "format": "ND", "ori_shape": (16, 1888, 11, 3), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (11, 11), (3, 3)]},
         {"shape": (-1, 1888, 11, 3), "dtype": "float32", "format": "ND", "ori_shape": (16, 1888, 11, 3), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (11, 11), (3, 3)]},
         {"shape": (-1, 1888, 11, 3), "dtype": "float32", "format": "ND", "ori_shape": (16, 1888, 11, 3), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (11, 11), (3, 3)]},
         {"shape": (-1, 1888, 11, 3), "dtype": "float32", "format": "ND", "ori_shape": (16, 1888, 11, 3), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (11, 11), (3, 3)]},
         {"shape": (-1, 1888, 11, 3), "dtype": "float32", "format": "ND", "ori_shape": (16, 1888, 11, 3), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (11, 11), (3, 3)]},
         {"shape": (-1, 1888, 11, 3), "dtype": "float32", "format": "ND", "ori_shape": (16, 1888, 11, 3), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (11, 11), (3, 3)]},
         {"shape": (-1, 1888, 11, 3), "dtype": "float32", "format": "ND", "ori_shape": (16, 1888, 11, 3), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (11, 11), (3, 3)]},
         {"shape": (-1, 1888, 11, 3), "dtype": "float32", "format": "ND", "ori_shape": (16, 1888, 11, 3), "ori_format": "ND",
          "range": [(16, 16), (1888, 1888), (11, 11), (3, 3)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_8",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
         "expect": "success",
         "support_expect": True}

case9 = {"params": [
         {"shape": (-1), "dtype": "float16", "format": "ND", "ori_shape": (3), "ori_format": "ND",
          "range": [(3, 3)]},
         {"shape": (-1), "dtype": "float16", "format": "ND", "ori_shape": (3), "ori_format": "ND",
          "range": [(3, 3)]},
         {"shape": (-1), "dtype": "float16", "format": "ND", "ori_shape": (3), "ori_format": "ND",
          "range": [(3, 3)]},
         {"shape": (-1), "dtype": "float16", "format": "ND", "ori_shape": (3), "ori_format": "ND",
          "range": [(3, 3)]},
         {"shape": (-1), "dtype": "float16", "format": "ND", "ori_shape": (3), "ori_format": "ND",
          "range": [(3, 3)]},
         {"shape": (-1), "dtype": "float16", "format": "ND", "ori_shape": (3), "ori_format": "ND",
          "range": [(3, 3)]},
         {"shape": (-1), "dtype": "float16", "format": "ND", "ori_shape": (3), "ori_format": "ND",
          "range": [(3, 3)]},
         {"shape": (-1), "dtype": "float16", "format": "ND", "ori_shape": (3), "ori_format": "ND",
          "range": [(3, 3)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_9",
         "expect": "failed",
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case3)
ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case4)
ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case5)
ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case6)
ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case7)
ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case8)
ut_case.add_case(["Ascend910A", "Ascend710A", "Ascend310"], case9)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710A", "Ascend310"])
