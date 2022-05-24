#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_dynamic_relu_v2
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("ReluV2", "impl.dynamic.relu_v2", "relu_v2")

case1 = {
    "params": [{
        "shape": (1, -1),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 8),
        "ori_format": "NC1HWC0",
        "range": [(1, 1), (1, None)]
    }, {
        "shape": (1, -1),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (1, 8),
        "ori_format": "NC1HWC0",
        "range": [(1, 1), (1, None)]
    }, {
        "shape": (1, -1),
        "dtype": "uint8",
        "format": "ND",
        "ori_shape": (1, 1),
        "ori_format": "ND",
        "range": [(1, 1), (1, None)]
    }],
    "case_name": "relu_v2_float16",
    "expect": "success"
}

case2 = {
    "params": [{
        "shape": (1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1, 8),
        "ori_format": "NC1HWC0",
        "range": [(1, 1), (1, None)]
    }, {
        "shape": (1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1, 8),
        "ori_format": "NC1HWC0",
        "range": [(1, 1), (1, None)]
    }, {
        "shape": (1, -1),
        "dtype": "uint8",
        "format": "ND",
        "ori_shape": (1, 1),
        "ori_format": "ND",
        "range": [(1, 1), (1, None)]
    }],
    "case_name": "relu_v2_float32",
    "expect": "success"
}

case3 = {
    "params": [{
        "shape": (1, -1),
        "dtype": "int8",
        "format": "ND",
        "ori_shape": (1, 8),
        "ori_format": "NC1HWC0",
        "range": [(1, 1), (1, None)]
    }, {
        "shape": (1, -1),
        "dtype": "int8",
        "format": "ND",
        "ori_shape": (1, 8),
        "ori_format": "NC1HWC0",
        "range": [(1, 1), (1, None)]
    }, {
        "shape": (1, -1),
        "dtype": "uint8",
        "format": "ND",
        "ori_shape": (1, 1),
        "ori_format": "ND",
        "range": [(1, 1), (1, None)]
    }],
    "case_name": "relu_v2_int8",
    "expect": "success"
}


def test_op_get_op_support_info(test_arg):
    from impl.dynamic.relu_v2 import get_op_support_info
    get_op_support_info(
        {
            "shape": (2, 2, 4, 8, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (2, 2, 4, 8, 16),
            "ori_format": "NC1HWC0"
        }, {
            "shape": (2, 2, 4, 8, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (2, 2, 4, 8, 16),
            "ori_format": "NC1HWC0"
        }, {
            "shape": (2, 2, 4, 8, 2),
            "dtype": "uint8",
            "format": "ND",
            "ori_shape": (2, 2, 4, 8, 2),
            "ori_format": "ND"
        }, "get_op_support_info_case1")


ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_cust_test_func(test_func=test_op_get_op_support_info)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
