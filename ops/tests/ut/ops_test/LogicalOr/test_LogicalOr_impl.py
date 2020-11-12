#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = BroadcastOpUT("LogicalOr", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["bool"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================
def calc_expect_func(x1, x2, y):
    x1_shape = x1.get("shape")
    x2_shape = x2.get("shape")
    x1_value = x1.get("value")
    x2_value = x2.get("value")

    output_data = np.logical_or(x1_value, x2_value)
    result = output_data.astype(np.int8)
    return (result,)

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input", "value_range":[-1,1.1]},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input", "value_range":[-1,1.1]},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "int8", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input", "value_range":[-1,1.1]},
                                              {"shape": (2, 16, 32), "dtype": "int8", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input", "value_range":[-1,1.1]},
                                              {"shape": (2, 16, 32), "dtype": "int8", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 24, 1, 256), "dtype": "int8", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input", "value_range":[-1,1.1]},
                                              {"shape": (1, 24, 1, 256), "dtype": "int8", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input", "value_range":[-1,1.1]},
                                              {"shape": (1, 24, 1, 256), "dtype": "int8", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
