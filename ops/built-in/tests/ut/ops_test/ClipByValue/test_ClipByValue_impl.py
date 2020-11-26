#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("ClipByValue", None, None)

case1 = {"params": [{"shape": (5, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (5, 1),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (5, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (5, 1),"ori_format": "NHWC"}],
         "case_name": "broadcast_to_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 11),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (2, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 11),"ori_format": "NHWC"}],
         "case_name": "broadcast_to_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 16),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 16),"ori_format": "NHWC"}],
         "case_name": "broadcast_to_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 32),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (32, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 32),"ori_format": "NHWC"}],
         "case_name": "broadcast_to_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1321, 73), "dtype": "float16", "format": "NHWC", "ori_shape": (1321, 73),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (1321, 73), "dtype": "float16", "format": "NHWC", "ori_shape": (1321, 73),"ori_format": "NHWC"}],
         "case_name": "broadcast_to_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (5, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 1),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (5, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 1),"ori_format": "NHWC"}],
         "case_name": "broadcast_to_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)

def calc_expect_func(x1, x2, x3, y):
    min = np.minimum(x1['value'], x3['value'])
    res = np.maximum(min, x2['value'])
    return res

precision_case1 = {"params": [{"shape": (5, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 1),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (5, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 1),"ori_format": "NHWC", "param_type": "output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case2 = {"params": [{"shape": (2, 11), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 11),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (2, 11), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 11),"ori_format": "NHWC", "param_type": "output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case3 = {"params": [{"shape": (32, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 32),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (32, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 32),"ori_format": "NHWC", "param_type": "output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}
precision_case4 = {"params": [{"shape": (512, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (512, 1),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC", "param_type": "input"},
                              {"shape": (512, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (512, 1),"ori_format": "NHWC", "param_type": "output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)
ut_case.add_precision_case("Ascend910", precision_case3)
ut_case.add_precision_case("Ascend910", precision_case4)
