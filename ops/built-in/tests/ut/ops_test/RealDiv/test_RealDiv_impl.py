#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = BroadcastOpUT("RealDiv", None, None)


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
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x1, x2, y):
    res =  x1['value'] / x2['value']
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

# pylint: disable=unused-argument
# ut_case.add_test_cfg_cov_case("all")
def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.real_div import op_select_format
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     "test_real_div_op_select_format_1")
    op_select_format({"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     "test_real_div_op_select_format_2")
    op_select_format({"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     "test_real_div_op_select_format_3")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     "test_real_div_op_select_format_4")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     "test_real_div_op_select_format_5")
    op_select_format({"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     "test_real_div_op_select_format_6")
    op_select_format({"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     "test_real_div_op_select_format_7")
    op_select_format({"shape": (16, 16, 32, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 32, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 32, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 32, 16),
                      "ori_format": "NCHW"},
                     "test_real_div_op_select_format_8")
    op_select_format({"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,), "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     "test_real_div_op_select_format_9")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     "test_real_div_op_select_format_10")
    op_select_format({"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     "test_real_div_op_select_format_11")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     "test_real_div_op_select_format_12")
ut_case.add_cust_test_func(test_func=test_op_select_format)
