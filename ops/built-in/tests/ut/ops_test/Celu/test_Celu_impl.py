#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = ElementwiseOpUT("Celu", None, None)


# ============ auto gen ["Ascend310"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (1,))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (16, 2, 4, 32))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (512, 1024))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (2, 1024))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (4096, 1024))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (32, 128, 1024))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (100, 100))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (1, 512, 1))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (1, 16, 512, 512))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (9973, 1))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (1024, 1024, 256))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (11, 33))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (10, 12))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16", "float32"], (10, 13))

# ============ auto gen ["Ascend310"] test cases end =================

def calc_expect_func(x, y):
    input_Arr = x['value'].astype(np.float32)
    result = np.where(input_Arr > 0, input_Arr, (np.exp(input_Arr) - 1)).astype(y['dtype'])
    return result



ut_case.add_precision_case("Ascend310", {"params": [{"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                         })
ut_case.add_precision_case("Ascend310", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                         })
ut_case.add_precision_case("Ascend310", {"params": [{"shape": (16, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (16, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                         })
ut_case.add_precision_case("Ascend310", {"params": [{"shape": (16, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (16, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                         })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (11, 33), "dtype": "float32", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND", "param_type": "input","value_range":[-1,1]},
                                                    {"shape": (11, 33), "dtype": "float32", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                         })
ut_case.add_precision_case("Ascend310", {"params": [{"shape": (12, 44), "dtype": "float32", "format": "ND", "ori_shape": (12, 44),"ori_format": "ND", "param_type": "input","value_range":[-10,10]},
                                                    {"shape": (12, 44), "dtype": "float32", "format": "ND", "ori_shape": (12, 44),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                         })

