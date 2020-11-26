#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = ElementwiseOpUT("FastGelu", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1,))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 4, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (512, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (2, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (4096, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (32, 128, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (100, 100))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1))

def calc_expect_func(x, y):
    attr = 1.702
    attr_opp = 0 - attr
    attr_half = attr / 2
    abs_x = np.abs(x['value'])
    mul_abs_x = abs_x * attr_opp
    exp_abs_x = np.exp(mul_abs_x)
    div_down = exp_abs_x + 1.0

    pn_x = x['value'] - abs_x
    mul_pn_x = pn_x * attr_half
    exp_pn_x = np.exp(mul_pn_x)
    div_up = x['value'] * exp_pn_x
    res = div_up / div_down
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
