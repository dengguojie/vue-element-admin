#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = BroadcastOpUT("DivNoNan", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 32), (16, 2, 32))
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

def calc_expect_func(x1, x2, y):
    output = np.divide(x1['value'], x2['value'])
    where_are_inf = np.isinf(output)
    output[where_are_inf] = 0
    if y['dtype'] in ("int32","int8","uint8",):
        output = np.floor(output)
    return output

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
