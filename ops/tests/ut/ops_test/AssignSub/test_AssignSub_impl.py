#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AssignSub", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "assign_sub_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

def calc_expect_func(x1, x2, y):
    res =  x2['value'] - x1['value']
    return res.astype(y['dtype'])

# ut_case.add_precision_case("all", {"params": [{"shape": (15,32), "dtype": "float16", "format": "ND", "ori_shape": (15,32),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (15,32), "dtype": "float16", "format": "ND", "ori_shape": (15,32),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (15,32), "dtype": "float16", "format": "ND", "ori_shape": (15,32),"ori_format": "ND", "param_type": "output"},
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })
# ut_case.add_precision_case("all", {"params": [{"shape": (16, 2 ,32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2 ,32),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16, 2 ,32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2 ,32),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16, 2 ,32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2 ,32),"ori_format": "ND", "param_type": "output"},
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })
# ut_case.add_precision_case("all", {"params": [{"shape": (5, 13, 64), "dtype": "float16", "format": "ND", "ori_shape": (5, 13, 64),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (5, 13, 64), "dtype": "float16", "format": "ND", "ori_shape": (5, 13, 64),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (5, 13, 64), "dtype": "float16", "format": "ND", "ori_shape": (5, 13, 64),"ori_format": "ND", "param_type": "output"},
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })
