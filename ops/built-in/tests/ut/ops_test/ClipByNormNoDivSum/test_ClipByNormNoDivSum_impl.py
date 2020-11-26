#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("ClipByNormNoDivSum", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "clip_by_norm_no_div_sum_compute_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "clip_by_norm_no_div_sum_compute_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "clip_by_norm_no_div_sum_compute_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "clip_by_norm_no_div_sum_compute_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "clip_by_norm_no_div_sum_compute_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

def calc_expect_func(x, greater_zeros, select_ones, maximum_ones, y):
    greater_result = x['value'] > greater_zeros['value']
    select_result = np.where(greater_result, x['value'], select_ones['value'])
    sqrt_result = np.sqrt(select_result)
    select1_result = np.where(greater_result, sqrt_result, x['value'])
    res = np.maximum(select1_result, maximum_ones['value'])
    return res

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 2, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (32, 2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 4),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)})
