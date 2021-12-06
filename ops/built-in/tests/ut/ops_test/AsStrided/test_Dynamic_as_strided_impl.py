#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import sys
from impl.util.platform_adapter import tbe_context
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

ut_case = OpUT("AsStrided", "impl.dynamic.as_strided", "as_strided")

def calc_sizeof(x):
    dtype = x.get("dtype")
    if dtype == "int64" or dtype == "uint64":
        return 8
    if dtype == "int32" or dtype == "uint32"  or dtype == "float32":
        return 4
    if dtype == "int16" or dtype == "uint16"  or dtype == "float16":
        return 2 
    if dtype == "int8":
        return 1 
    return 0


def calc_expect_func(x, size, stride, storage_offset, actual):
    expect = np.lib.stride_tricks.as_strided(x.get("value"), size.get("value"), stride.get("value") * calc_sizeof(x))
    print("------------------actual---------------------")
    print(actual.get("value"))
    print("------------------expect---------------------")
    print(expect)
    return (expect,)


def gen_as_strided_case(dynamic_input_shapes, ori_input_shapes, dtype, size, stride,
                       case_name_val, expect):
    inputs = (
        {"shape": dynamic_input_shapes,
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": "ND",
         "format": "ND",
         'range': [[1, 100000]] * len(dynamic_input_shapes)},
        {"shape": [len(size)],
         "dtype": "int32",
         "ori_shape": [len(size)],
         "ori_format": "ND",
         "format": "ND",
         'range': [[1, 100000]] * len(size)},
        {"shape": [len(size)],
         "dtype": "int32",
         "ori_shape": [len(size)],
         "ori_format": "ND",
         "format": "ND",
         'range': [[1, 100000]] * len(size)},
        {"shape": [1],
         "dtype": "int32",
         "ori_shape": [1],
         "ori_format": "ND",
         "format": "ND",
         'range': [[1, 100000]]},
    )

    outputs = (
        {"shape": size,
         "dtype": dtype,
         "ori_shape": size,
         "ori_format": "ND",
         "format": "ND",
         'range': [[1, 100000]] * len(size)},
    )
    return {"params": [inputs[0],
                       inputs[1],
                       inputs[2],
                       inputs[3],
                       outputs[0],
                       ],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A"],
                 gen_as_strided_case((-1, -1, -1, -1, -1),
                                    (2, 11, 2673, 11, 16, 16),
                                    "float32", (2,3,4), (1,1,1), "case_1", "success"))
ut_case.add_case(["Ascend910A"],
                 gen_as_strided_case((-1, -1, -1, -1, -1),
                                    (2, 11, 2673, 11, 16, 16),
                                    "int8", (2,3,4), (1,1,1), "case_2", "success"))
ut_case.add_case(["Ascend910A"],
                 gen_as_strided_case((-1, -1, -1, -1, -1),
                                    (2, 11, 2673, 11, 16, 16),
                                    "float16", (2,3,4), (1,1,1), "case_3", "success"))
