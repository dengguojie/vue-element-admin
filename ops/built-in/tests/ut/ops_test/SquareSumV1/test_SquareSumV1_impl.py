#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("SquareSumV1", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    [1,2]],
         "case_name": "square_sum_v1_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    [0,1]],
         "case_name": "square_sum_v1_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "square_sum_v1_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "square_sum_v1_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    [0,1]],
         "case_name": "square_sum_v1_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (3,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    [0,1]],
         "case_name": "square_sum_v1_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)

def calc_expect_func(x, y, axis):
    square = x['value'] * x['value']
    res = square.sum(axis=tuple(axis))
    res = res.astype(y['dtype'])
    return res

precision_case1 = {"params": [{"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","param_type":"output"}, [0]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND","param_type":"output"},[1]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (16,2,32), "dtype": "float16", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND","param_type":"output"},[1,2]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case4 = {"params": [{"shape": (16,2,4,32), "dtype": "float16", "format": "ND", "ori_shape": (16,2,4,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (32,), "dtype": "float16", "format": "ND", "ori_shape": (32,),"ori_format": "ND","param_type":"output"},[0,1,2]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}


ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)
ut_case.add_precision_case("Ascend910", precision_case3)
ut_case.add_precision_case("Ascend910", precision_case4)


def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.square_sum_v1 import op_select_format
    op_select_format({"shape": (1, 1, 16, 16), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 16),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 16), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 16),
                      "ori_format": "HWCN"},
                     [0,1,2,3],
                     attr2=True,
                     kernel_name="test_square_sum_v1_op_select_format_1")
    op_select_format({"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16),
                      "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16),
                      "ori_format": "ND"},
                     None,
                     attr2=True,
                     kernel_name="test_square_sum_v1_op_select_format_2")


ut_case.add_cust_test_func(test_func=test_op_select_format)