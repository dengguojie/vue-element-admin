#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = ElementwiseOpUT("Asin", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (1,))
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (16, 2, 32))

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x, output):
    shape_x = x.get("shape")
    x_value = x.get("value")
    output = np.arcsin(x_value)
    return output

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

