#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("ReluV2", None, None)

case1 = {"params": [{"shape": (1,8), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,8),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "relu_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,8), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,8),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "relu_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,2,4,8), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,8),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "relu_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

def calc_expect_func(x, y, mask):
    input_list = x['value']
    shape = x['shape']
    shape_mask = []
    shape_sum = 1
    for i, _ in enumerate(shape):
        if i == (len(shape)-1):
            shape_mask.append(shape[i]//8)
        else:
            shape_mask.append(shape[i])
        shape_sum *= shape[i]

    result = np.maximum(input_list, 0)
    input_list = list(input_list.reshape((shape_sum, 1)))
    mask = []
    num = 0
    sum_num = 0
    for i in input_list:
        if i > 0:
            sum_num += np.power(2, num)
        num += 1
        if num == 8:
            num = 0
            mask.append(sum_num)
            sum_num = 0
    mask = np.array(mask)
    mask = mask.reshape(shape_mask)
    return result, mask

ut_case.add_precision_case("all", {"params": [{"shape": (1, 8), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND", "param_type": "input","value_range":[-1, 1]},
                                              {"shape": (1, 8), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND", "param_type": "output"},
                                              {"shape": (1, 1), "dtype": "uint8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input","value_range":[-1, 1]},
                                              {"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              {"shape": (3, 16, 4), "dtype": "uint8", "format": "ND", "ori_shape": (3, 16, 4),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 3, 100, 8), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 8),"ori_format": "ND", "param_type": "input","value_range":[-1, 1]},
                                              {"shape": (1, 3, 100, 8), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 8),"ori_format": "ND", "param_type": "output"},
                                              {"shape": (1, 3, 100, 1), "dtype": "uint8", "format": "ND", "ori_shape": (1, 3, 100, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
