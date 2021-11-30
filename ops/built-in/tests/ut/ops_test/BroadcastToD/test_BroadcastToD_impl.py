#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("BroadcastToD", None, None)

case1 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    (1, )],
         "case_name": "broadcast_to_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1, 3), "dtype": "float32", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float32", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    (3, 3)],
         "case_name": "broadcast_to_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1, 3), "dtype": "int32", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "int32", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    (3, 3)],
         "case_name": "broadcast_to_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

def calc_expect_func(x, y, shape):
    out = np.broadcast_to(x['value'], shape)
    return out

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              (1, 1)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},
                                              (16, 32)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (16, 1), "dtype": "float16", "format": "ND", "ori_shape": (16, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},
                                              (16, 32)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
