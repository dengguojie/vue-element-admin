#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = ElementwiseOpUT("Inv", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 4, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (512, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (2, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (4096, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (32, 128, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (100, 100))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 512, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 16, 512, 512))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (9973, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1024, 1024, 256))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (11, 33))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (10, 12))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (10, 13))

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x, y):
    one = np.ones(x['shape']).astype(x['dtype'])
    res = one / x['value']
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
