#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("Gather", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "int32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "gather_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "int32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "gather_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "int32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "gather_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "int32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "gather_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "int32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "gather_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

def calc_expect_func(x, indice, y):
    res = x['value'][indice['value']]
    return res

precision_case1 = {"params": [{"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND","param_type":"input", "value_range":[0,15]},
                              {"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (16,2,32), "dtype": "float16", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND","param_type":"input", "value_range":[0,15]},
                              {"shape": (16,2,32), "dtype": "float16", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (512,1024), "dtype": "float16", "format": "ND", "ori_shape": (512,1024),"ori_format": "ND","param_type":"input"},
                              {"shape": (512,), "dtype": "int32", "format": "ND", "ori_shape": (512,),"ori_format": "ND","param_type":"input", "value_range":[0,500]},
                              {"shape": (512,1024), "dtype": "float16", "format": "ND", "ori_shape": (512,1024),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}


def test_get_op_support_info(test_arg):
    from impl.dynamic.gather import get_op_support_info
    get_op_support_info({"shape": [-2], "dtype": "int8", "format": "ND", "ori_shape": [20, 28], "ori_format": "ND"},
                     {"shape": [200], "dtype": "int32", "format": "ND", "ori_shape": [200], "ori_format": "ND"},
                     {"shape": [200, 28], "dtype": "int8", "format": "NCHW", "ori_shape": [200, 28],"ori_format": "ND"})
    get_op_support_info({"shape": [20], "dtype": "float16", "format": "ND", "ori_shape": [-2], "ori_format": "ND"},
                     {"shape": [10], "dtype": "int32", "format": "ND", "ori_shape": [10], "ori_format": "ND"},
                     {"shape": [10], "dtype": "float16", "format": "NCHW", "ori_shape": [10], "ori_format": "ND"})
    get_op_support_info({"shape": [30, 5, 61], "dtype": "int32", "format": "ND", "ori_shape": [30, 5, 61],
                     "ori_format": "ND"},
                     {"shape": [10], "dtype": "int32", "format": "ND", "ori_shape": [-2], "ori_format": "ND"},
                     {"shape": [10, 5, 61], "dtype": "int32", "format": "NCHW", "ori_shape": [10, 5, 61],
                      "ori_format": "ND"})


ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)
ut_case.add_precision_case("Ascend910", precision_case3)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)
