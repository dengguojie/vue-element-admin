#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = ElementwiseOpUT("Tanh", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1,))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 4, 32))

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x, output):
    shape_x = x.get("shape")
    x_value = x.get("value")

    dtype = x["dtype"]
    if dtype == "fp16" or dtype == "float16":
        s_type = np.float16
    elif dtype == "fp32" or dtype == "float32":
        s_type = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    output = np.tanh(x_value).astype(s_type)
    return output

ut_case.add_precision_case(["Ascend910"], {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case(["Ascend910"], {"params": [{"shape": (3, 16, 32), "dtype": "float32", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (3, 16, 32), "dtype": "float32", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case(["Ascend910"], {"params": [{"shape": (1, 3, 100, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 3, 100, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
