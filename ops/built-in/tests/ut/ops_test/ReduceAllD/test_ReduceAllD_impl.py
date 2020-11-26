#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ReduceOpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = ReduceOpUT("ReduceAllD", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1,), (0,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1,), 0, False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1, 1), (1,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1, 1), (1,), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (101, 10241), (-1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (101, 10241), (-1, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1023*255, ), (-1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1023*255, ), (-1, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (51, 101, 1023), (1, 2), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (51, 101, 1023), (1, 2), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (51, 101, 1023), (1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (51, 101, 1023), (1, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (51, 101, 1023), (0, 1, 2), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (51, 101, 1023), (0, 1, 2), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (99991, 10), (0, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (99991, 10), (0, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1, 99991), (1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1, 99991), (1, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1, 99991, 10), (1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["bool"], (1, 99991, 10), (1, ), False)

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x, y, axes):
    x_shape = x["shape"]
    x_format = x["format"]
    input_data = x["value"]

    data_abs = np.abs(input_data)
    result = np.min(data_abs, axes)

    return result

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
                                              (1,)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "int8", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, ), "dtype": "int8", "format": "ND", "ori_shape": (2, ),"ori_format": "ND", "param_type": "output"},
                                              (1, 2)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 24, 1, 256), "dtype": "int8", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (256, ), "dtype": "int8", "format": "ND", "ori_shape": (256, ),"ori_format": "ND", "param_type": "output"},
                                              (0, 1, 2)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)



