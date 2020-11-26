#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = BroadcastOpUT("LeakyReluGrad", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (100, 100), (100, 100))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1), (1,))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (11, 33), (11, 33))


# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(g, x, y, negative_slope = 0):
    dtype = x['dtype']
    if dtype == "float32":
        help_min = 2 ** (-126)
        help_rec_one = 2 ** (38)
        help_rec_sec = 2 ** (44)
    elif dtype == "float16":
        help_min = 2 ** (-24)
        help_rec_one = 2 ** (12)
        help_rec_sec = 2 ** (12)
    tmp_min_x = np.minimum(x['value'], help_min)
    tmp_max_x = np.maximum(tmp_min_x, 0)
    tmp_mul_x = tmp_max_x * help_rec_one

    if dtype == "float32":
        tmp_mul_x = tmp_mul_x * help_rec_sec

    result_tmp_right = tmp_mul_x * help_rec_sec
    result_abs = np.abs(result_tmp_right - 1.0)
    result_tmp_left = result_abs * negative_slope
    res = g['value'] * (result_tmp_left + result_tmp_right)

    return res

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

