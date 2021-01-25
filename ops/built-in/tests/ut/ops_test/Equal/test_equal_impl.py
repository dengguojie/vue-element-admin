#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("Equal", "impl.equal", "equal")

def calc_expect_func(x1, x2, y):
    res = np.equal(x1['value'], x2['value']).astype(np.int8)
    return res

def gen_equal_case(shape_x, shape_y, dtype_val, expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "param_type":"input"},
        {"ori_shape": shape_y, "shape": shape_y, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "param_type":"input"},
        {"ori_shape": shape_x, "shape": shape_y, "ori_format": "ND",
         "format": "ND", "dtype": "int8", "param_type":"output"}],
        "expect": expect,
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_precision_case("all",
                           gen_equal_case((16,), (16,), "int32", "success"))
ut_case.add_precision_case("all",
                           gen_equal_case((16,), (16,), "float32", "success"))
ut_case.add_precision_case("all",
                           gen_equal_case((3,), (3,), "bool", "success"))

