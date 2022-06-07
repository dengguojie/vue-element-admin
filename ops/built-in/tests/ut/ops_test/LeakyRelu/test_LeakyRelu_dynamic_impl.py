#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("LeakyRelu", "impl.dynamic.leaky_relu", "leaky_relu")


case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 100)]},
        0
    ],
    "case_name": "dynamic_leaky_relu_case1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
    }

case2 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 100)]},
        None
    ],
    "case_name": "dynamic_leaky_relu_case2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
    }


case3 = {
    "params": [
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND"},
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND"},
        None
    ],
    "case_name": "dynamic_leaky_relu_case3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
    }

ut_case.add_case(["Ascend910A", "Ascend310P3", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310P3", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend310P3", "Ascend310"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
