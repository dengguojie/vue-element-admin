#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ReduceOpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = ReduceOpUT("ReduceProdD", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1,), (0,), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1,), 0, False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1, 1), (1,), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1, 1), (1,), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (101, 10241), (-1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (101, 10241), (-1, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1023*255, ), (-1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1023*255, ), (-1, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (51, 101, 1023), (1, 2), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (51, 101, 1023), (1, 2), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (51, 101, 1023), (1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (51, 101, 1023), (1, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (51, 101, 1023), (0, 1, 2), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (51, 101, 1023), (0, 1, 2), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (99991, 10), (0, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (99991, 10), (0, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1, 99991), (1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1, 99991), (1, ), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1, 99991, 10), (1, ), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int8", "uint8"], (1, 99991, 10), (1, ), False)

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x, y, axes):
    x_shape = x["shape"]
    x_format = x["format"]
    input_data = x["value"]

    result = np.prod(input_data, axes)

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

from impl.reduce_prod_d import check_supported

def test_check_support(test_arg):
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (256, ), "dtype": "float32", "format": "ND", "ori_shape": (256, ),"ori_format": "ND", "param_type": "output"},
                                              (0, -1, 2))


ut_case.add_cust_test_func(test_func=test_check_support)


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

