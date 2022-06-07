#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("SigmoidCrossEntropyWithLogits", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)

def calc_expect_func(x1, x2, y):
    shape = x1['shape']
    zero = np.zeros(shape)
    input = x1['value']
    target = x2['value']
    sig = np.fmax(input, zero) - input*target + np.log(1.00000000 + np.exp(-np.abs(input)))
    return sig

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 16),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16, 64), "dtype": "float16", "format": "ND", "ori_shape": (16, 64),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 64), "dtype": "float16", "format": "ND", "ori_shape": (16, 64),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 64), "dtype": "float16", "format": "ND", "ori_shape": (16, 64),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (2, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 2, 32),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 2, 32),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 2, 32),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (4, 2, 64, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 2, 64, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (4, 2, 64, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 2, 64, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (4, 2, 64, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 2, 64, 16),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                         })




