#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("GatherV2D", None, None)

case1 = {"params": [{"shape": (30522, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    {"shape": (512,), "dtype": "int32", "format": "NHWC", "ori_shape": (512,),"ori_format": "NHWC"},
                    {"shape": (30522, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    0],
         "case_name": "gather_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (30522, 1024), "dtype": "float16", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    {"shape": (512,), "dtype": "int32", "format": "NHWC", "ori_shape": (512,),"ori_format": "NHWC"},
                    {"shape": (30522, 1024), "dtype": "float16", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    0],
         "case_name": "gather_v2_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (30522, 1024), "dtype": "int32", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    {"shape": (512,), "dtype": "int32", "format": "NHWC", "ori_shape": (512,),"ori_format": "NHWC"},
                    {"shape": (30522, 1024), "dtype": "int32", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    0],
         "case_name": "gather_v2_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (30522, 1024), "dtype": "uint8", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    {"shape": (512,), "dtype": "int32", "format": "NHWC", "ori_shape": (512,),"ori_format": "NHWC"},
                    {"shape": (30522, 1024), "dtype": "uint8", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    0],
         "case_name": "gather_v2_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (30522, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    {"shape": (1,), "dtype": "int32", "format": "NHWC", "ori_shape": (1,),"ori_format": "NHWC"},
                    {"shape": (30522, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (30522, 1024),"ori_format": "NHWC"},
                    0],
         "case_name": "gather_v2_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (1, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1),"ori_format": "NHWC"},
                    {"shape": (1,), "dtype": "int32", "format": "NHWC", "ori_shape": (1,),"ori_format": "NHWC"},
                    {"shape": (1, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1),"ori_format": "NHWC"},
                    0],
         "case_name": "gather_v2_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (1, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1),"ori_format": "NHWC"},
                    {"shape": (1,), "dtype": "float16", "format": "NHWC", "ori_shape": (1,),"ori_format": "NHWC"},
                    {"shape": (1, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1),"ori_format": "NHWC"},
                    0],
         "case_name": "gather_v2_d_7",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)


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


ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)
ut_case.add_precision_case("Ascend910", precision_case3)
