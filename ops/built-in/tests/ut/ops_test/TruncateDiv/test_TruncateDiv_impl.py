#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = BroadcastOpUT("TruncateDiv", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 4, 32), (16, 2, 4, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (512, 1024), (512, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (2, 1024), (2, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (4096, 1024), (4096, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (32, 128, 1024), (32, 128, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (100, 100), (100, 100))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 512, 1), (1,))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 16, 512, 512), (1, 1, 512, 512))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (9973, 1), (9973, 1))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1024, 1024, 256), (1024, 1024, 256))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x1, x2, y):
    x1_value = x1.get("value")
    x2_value = x2.get("value")
    input_dtype = x1.get("dtype")

    int_list = ("int32", "int8", "uint8")
    if input_dtype in int_list:

        res_div = x1_value/x2_value
        res_min_int = np.ceil(np.minimum(res_div, 0))
        res_max_int = np.floor(np.maximum(res_div, 0))
        res = res_min_int+res_max_int
    else:
        res = x1_value/x2_value

    return res.astype(y["dtype"])

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input", "value_range":[0,127]},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input", "value_range":[0,127]},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
