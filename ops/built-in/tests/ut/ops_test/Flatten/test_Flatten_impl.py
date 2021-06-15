#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("Flatten", None, None)

case1 = {"params": [{"shape": (255,8,33), "dtype": "float32", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    {"shape": (255,8,33), "dtype": "float32", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"}, 1],
         "case_name": "Flatten_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (255,8,33), "dtype": "float16", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    {"shape": (255,8,33), "dtype": "float16", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"}, 1],
         "case_name": "Flatten_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (4, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"}, 1],
         "case_name": "Flatten_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (4, 16, 64), "dtype": "int32", "format": "ND", "ori_shape": (4, 16, 64),"ori_format": "ND"},
                    {"shape": (4, 16, 64), "dtype": "int32", "format": "ND", "ori_shape": (4, 16, 64),"ori_format": "ND"}, 1],
         "case_name": "Flatten_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

# pylint: disable = unused-argument
def calc_expect_func(x, y, axis=1):
    res = x['value'].flatten()
    return res

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (4800, ), "dtype": "float16", "format": "ND", "ori_shape": (4800, ),"ori_format": "ND", "param_type": "output"}, 1
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 3, 100, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (4800, ), "dtype": "float32", "format": "ND", "ori_shape": (4800, ),"ori_format": "ND", "param_type": "output"}, 0
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 3, 100, 16), "dtype": "int8", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (4800, ), "dtype": "int8", "format": "ND", "ori_shape": (4800, ),"ori_format": "ND", "param_type": "output"}, 2
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 3, 100, 16), "dtype": "uint16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (4800, ), "dtype": "uint16", "format": "ND", "ori_shape": (4800, ),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 3, 100, 16), "dtype": "int64", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (4800, ), "dtype": "int64", "format": "ND", "ori_shape": (4800, ),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
