#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("NormalizeSum", None, None)

case1 = {"params": [{"shape": (1,2,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,2,4,16),"ori_format": "NCHW"},
                    {"shape": (1,2,4,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,2,4,16),"ori_format": "NCHW"}],
         "case_name": "normallize_sum_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (16,16,16,16),"ori_format": "NCHW"},
                    {"shape": (16,16,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (16,16,16,16),"ori_format": "NCHW"}],
         "case_name": "normallize_sum_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 2, 4, 16),"ori_format": "NCHW"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 2, 4, 16),"ori_format": "NCHW"}],
         "case_name": "normallize_sum_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (32, 2, 4, 16),"ori_format": "NCHW"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (32, 2, 4, 16),"ori_format": "NCHW"}],
         "case_name": "normallize_sum_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 2),"ori_format": "NCHW"},
                    {"shape": (1, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 2),"ori_format": "NCHW"}],
         "case_name": "normallize_sum_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

def test_get_op_support_info(test_arg):
    from impl.normalize_sum import get_op_support_info
    get_op_support_info({"shape": (15, 80, 38, 38), "dtype": "float16", "format": "NCHW","ori_shape":(15, 80, 2, 32), "ori_format":"NCHW"},
                        {"shape": (15, 80, 1, 38), "dtype": "float16", "format": "NCHW","ori_shape":(15, 80, 2, 32), "ori_format":"NCHW"},
                        True)

ut_case.add_cust_test_func(test_func=test_get_op_support_info)

def calc_expect_func(x1, y, across_spatial=True):
    x1_value = x1.get("value")
    data_format = x1["dtype"]

    x1_cast_sqr = x1_value**2

    if across_spatial:
        x1_cast_sqr_sum = np.sum(x1_cast_sqr, axis=(1, 2, 3))
    elif data_format == "NCHW":
        x1_cast_sqr_sum = np.sum(x1_cast_sqr, axis=1)
    elif data_format == "NHWC":
        x1_cast_sqr_sum = np.sum(x1_cast_sqr, axis=3)

    return x1_cast_sqr_sum.astype(np.float32)

ut_case.add_precision_case("all", {"params": [{"shape": (2, 3, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,),"ori_format": "NCHW", "param_type": "output"},
                                              True],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 3, 2, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,),"ori_format": "NCHW", "param_type": "output"},
                                              True],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 2, 3, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 3, 3),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,),"ori_format": "NCHW", "param_type": "output"},
                                              True],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

