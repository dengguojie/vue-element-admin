#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("DropOutDoMask", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "drop_out_do_mask_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (32,), "dtype": "float32", "format": "ND", "ori_shape": (32,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "drop_out_do_mask_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "drop_out_do_mask_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (512, ), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "drop_out_do_mask_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (16, ), "dtype": "float32", "format": "ND", "ori_shape": (16, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "drop_out_do_mask_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (3333333,), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (416672, ), "dtype": "float32", "format": "ND", "ori_shape": (16, ),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (3333333,), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "drop_out_do_mask_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend710", "Ascend910A"], case6)

def revert_bit(n):
    result = 0
    for i in range(8):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result

def revert_array_bit(arr):
    res = []
    for item in arr.flatten():
        res.append(revert_bit(item))
    return np.array(res, dtype=np.uint8).reshape(arr.shape)

def calc_expect_func(x1, x2, x3, y):
    shape_x = x1['shape']
    keep_prop = x3['value']
    input_x = x1['value']
    mask = x2['value']
    x_scale = input_x * (1.0 / keep_prop)
    mask = revert_array_bit(mask)
    mask_dtype = np.unpackbits(mask, axis=-1).astype(x1['dtype'])

    size_x = 1
    x_scale = x_scale.flatten()
    for i in shape_x:
        size_x = size_x * i
    expect = np.zeros(size_x).astype(y['dtype'])
    for i in range(size_x):
        expect[i] = x_scale[i] * mask_dtype[i]
    return expect.reshape(y['shape'])

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, ), "dtype": "uint8", "format": "ND", "ori_shape": (16, ),"ori_format": "ND", "param_type": "input","value_range":[0,255]},
                                                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (512, ), "dtype": "uint8", "format": "ND", "ori_shape": (16, ),"ori_format": "ND", "param_type": "input","value_range":[0,255]},
                                                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
