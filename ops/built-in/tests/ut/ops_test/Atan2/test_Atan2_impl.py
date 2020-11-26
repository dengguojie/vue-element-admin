#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = BroadcastOpUT("Atan2", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (16, 32), (16, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 32), (16, 2, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 4, 32), (16, 2, 4, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (512, 1024), (512, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (2, 1024), (2, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (4096, 1024), (4096, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (32, 128, 1024), (32, 128, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (100, 100), (100, 100))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1), (1,))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1, 16, 512, 512), (1, 1, 512, 512))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (9973, 1), (9973, 1))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1024, 1024, 256), (1024, 1024, 256))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (11, 33), (11, 33))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (10, 12), (10, 11), expect=RuntimeError)
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x1, x2, output):
    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    x1_value = x1.get("value")
    x2_value = x2.get("value")
    output = np.arctan2(x1_value, x2_value)
    return output

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 3, 10, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 3, 10, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 3, 10, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 3, 10, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 3, 10, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 3, 10, 16),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

