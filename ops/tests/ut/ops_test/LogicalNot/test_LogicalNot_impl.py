#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = ElementwiseOpUT("LogicalNot", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (1,))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (16, 2, 4, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (512, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (2, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (4096, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (32, 128, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (100, 100))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (1, 512, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (1, 16, 512, 512))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (9973, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (1024, 1024, 256))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (11, 33))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (10, 12))
ut_case.add_elewise_case_simple(["Ascend910"], ["int8"], (10, 13))

# ============ auto gen ["Ascend910"] test cases end =================


def calc_expect_func(x, y):
    x_shape = x.get("shape")
    x_value = x.get("value")

    ones = np.ones((1,)).astype(np.int8)
    result = ones-x_value
    return result

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "int8", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16, 32), "dtype": "int8", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 24, 1, 256), "dtype": "int8", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "int8", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/home/maying/.mindstudio/huawei/adk/1.76.T1.0.B010/toolkit/tools/simulator")
