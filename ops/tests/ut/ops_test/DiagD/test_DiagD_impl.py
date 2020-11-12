#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("DiagD", None, None)

case1 = {"params": [{"shape": (3,), "dtype": "float32", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    {"shape": (3,3), "dtype": "float32", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"},
                    {"shape": (3,3), "dtype": "float32", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"}],
         "case_name": "diag_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "diag_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32,), "dtype": "float32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, 32), "dtype": "float32", "format": "ND", "ori_shape": (32, 32),"ori_format": "ND"},
                    {"shape": (32, 32), "dtype": "float32", "format": "ND", "ori_shape": (32, 32),"ori_format": "ND"}],
         "case_name": "diag_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (128, ), "dtype": "float16", "format": "ND", "ori_shape": (128, ),"ori_format": "ND"},
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND"},
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND"}],
         "case_name": "diag_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "diag_d_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

def calc_expect_func(x1, x2, output):
    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    x1_value = x1.get("value")
    x2_value = x2.get("value")

    dtype = x1["dtype"]
    dtype = x2["dtype"]
    if dtype == "fp16" or dtype == "float16":
        s_type = np.float16
    elif dtype == "fp32" or dtype == "float32":
        s_type = np.float32
    elif dtype == "int32":
        s_type = np.int32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    output = x1_value*x2_value
    return output

ut_case.add_precision_case("all", {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (3,),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 16), "dtype": "int32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 16), "dtype": "int32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
