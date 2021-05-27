"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

BinaryCrossEntropyGrad ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("BinaryCrossEntropyGrad", None, None)

case1 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    "mean"
                    ],
         "case_name": "BinaryCrossEntropyGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    "sum"
                    ],
         "case_name": "BinaryCrossEntropyGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    None,
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    "none"
                    ],
         "case_name": "BinaryCrossEntropyGrad_3",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
#ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)

def calc_expect_func(x, target, dy, weight, y, reduction):
    shape = x['shape']
    reduce_elts = 1.0
    for i in shape:
        reduce_elts *= i
    cof = reduce_elts**(-1)

    dx = (x['value'] - target['value']) / np.maximum((x['value'] * (1 - x['value'])), 1e-12)
    if weight is not None:
        dx = dx * weight['value']
    res = dy['value'] * dx
    if reduction == "mean":
        dx *= cof
    return res

# ut_case.add_precision_case("all", {"params": [{"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "output"},
#                                               "none"],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
#                                    })
# ut_case.add_precision_case("all", {"params": [{"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float32", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "output"},
#                                               "sum"],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
#                                    })
# ut_case.add_precision_case("all", {"params": [{"shape": (16,8,35), "dtype": "float16", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float16", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float16", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float16", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (16,8,35), "dtype": "float16", "format": "ND", "ori_shape": (16,8,35),"ori_format": "ND", "param_type": "output"},
#                                               "none"],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
#                                    })
