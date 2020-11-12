#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = ElementwiseOpUT("ZerosLike", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1,))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 4, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (512, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (2, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (4096, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (32, 128, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (100, 100))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 512, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 16, 512, 512))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (9973, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1024, 1024, 256))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (11, 33))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (10, 12))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (10, 13))

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x, y):
    zeros = np.zeros(x['shape']).astype(x['dtype'])
    # res = np.zeros_like(x)
    # print("expres",res)
    return zeros

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 16, 16), "dtype": "int32", "format": "ND", "ori_shape": (1, 16, 16),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (1, 16, 16), "dtype": "int32", "format": "ND", "ori_shape": (1, 16, 16),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,4,4,8), "dtype": "int8", "format": "ND", "ori_shape": (1,4,4,8),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (1,4,4,8), "dtype": "int8", "format": "ND", "ori_shape": (1,4,4,8),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (11, 33), "dtype": "float16", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (11, 33), "dtype": "float16", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
