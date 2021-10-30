#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.non_zero_with_value import non_zero_with_value

ut_case = OpUT("NonZeroWithValue", "impl.non_zero_with_value", "non_zero_with_value")

case1 = {
    "params": [{
        "shape": (10, 20),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (10, 20),
        "ori_format": "ND"
    }, {
        "shape": (200),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (200),
        "ori_format": "ND"
    }, {
        "shape": (400),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (400),
        "ori_format": "ND"
    }, {
        "shape": (1),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (1),
        "ori_format": "ND"
    }, False],
    "case_name": "test_non_zero_with_value_001",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910"])
    exit(0)
