#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("Select", None, None)

def gen_select_case(shape_condtion, shape_var, dtype, expect, case_name_val,  bool_dtype="int8"):
    return {"params": [{"shape": shape_condtion, "dtype": bool_dtype, "ori_shape": shape_var, "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": "NCHW", "format": "NCHW"}],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_select_case((21,), (21,), "float16", "success", "select_1")
case2 = gen_select_case((21,), (21,), "int32", "success", "select_2")
case3 = gen_select_case((21,), (21,), "float32", "success", "select_3")
case4 = gen_select_case((100000,), (100000,), "float32", "success", "select_4")
case5 = gen_select_case((1,), (100000, 2147), "float16", "success", "select_5")
case6 = gen_select_case((1,), (20, 5, 5, 2, 2147, 20, 5, 10), "int32", "success", "select_6")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)


def calc_expect_func(condition, x1, x2, y):
    x1_shape = condition.get("shape")
    x2_shape = x1.get("shape")
    x3_shape = x2.get("shape")
    x1_value = condition.get("value")
    x2_value = x1.get("value")
    x3_value = x2.get("value")

    x1_value = x1_value.astype(np.bool)

    output_data = np.select(x1_value, x2_value, x3_value)
    #result = output_data.astype(np.int32)
    return output_data

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input", "value_range": [-1,2]},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (100000,), "dtype": "int8", "format": "ND", "ori_shape": (100000,),"ori_format": "ND", "param_type": "input", "value_range": [-1,2]},
                                              {"shape": (100000,), "dtype": "int8", "format": "ND", "ori_shape": (100000,),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (100000,), "dtype": "int8", "format": "ND", "ori_shape": (100000,),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (100000,), "dtype": "int8", "format": "ND", "ori_shape": (100000,),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 24, 1, 256), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 24, 1, 256),"ori_format": "NCHW", "param_type": "input", "value_range": [-1,2]},
                                              {"shape": (1, 24, 1, 256), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 24, 1, 256),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 24, 1, 256),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "int32", "format": "NCHW", "ori_shape": (1, 24, 1, 256),"ori_format": "NCHW", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)