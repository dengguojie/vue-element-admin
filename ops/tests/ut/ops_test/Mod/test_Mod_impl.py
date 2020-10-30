#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = BroadcastOpUT("Mod", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 32), (16, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 32), (16, 2, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 4, 32), (16, 2, 4, 32))
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
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (10, 12), (10, 11), expect=RuntimeError)
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x1, x2, res):
    x1_shape = x1.get("shape")
    x2_shape = x2.get("shape")
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
    elif dtype == "int8":
        s_type = np.int8
    elif dtype == "uint8":
        s_type = np.uint8
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    res = np.mod(x1_value, x2_value).astype(s_type)
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (4,4,32,2), "dtype": "int32", "format": "ND", "ori_shape": (4,4,32,2),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (4,4,32,2), "dtype": "int32", "format": "ND", "ori_shape": (4,4,32,2),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (4,4,32,2), "dtype": "int32", "format": "ND", "ori_shape": (4,4,32,2),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/home/maying/.mindstudio/huawei/adk/1.76.T1.0.B010/toolkit/tools/simulator")


