#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("GreaterEqual", None, None)

def gen_greater_equal_case(shape_x, shape_y, dtype):
    return {"params": [{"shape": shape_x, "dtype": dtype, "format": "ND", "ori_shape": shape_x,"ori_format": "ND", "param_type":"input"},
                       {"shape": shape_y, "dtype": dtype, "format": "ND", "ori_shape": shape_y,"ori_format": "ND", "param_type":"input"},
                       {"shape": shape_y, "dtype": "int8", "format": "ND", "ori_shape": shape_y,"ori_format": "ND", "param_type":"output"},
                       ],
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}

def calc_expect_func(x, y, output):
    res = np.greater_equal(x['value'], y['value']).astype(np.int8)
    return res

ut_case.add_precision_case("Ascend910A", gen_greater_equal_case((28, 28), (28, 28), "float32"))
ut_case.add_precision_case("Ascend910A", gen_greater_equal_case((1, 2), (1, 2), "float32"))
ut_case.add_precision_case("Ascend910A", gen_greater_equal_case((32, 2, 7), (32, 2, 7), "float32"))
ut_case.add_precision_case("Ascend910A", gen_greater_equal_case((10, 11), (10, 11), "float32"))

