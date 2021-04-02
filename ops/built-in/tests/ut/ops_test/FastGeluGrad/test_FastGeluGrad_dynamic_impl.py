#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
test for FastGeluGrad
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("FastGeluGrad", "impl.dynamic.fast_gelu_grad", "fast_gelu_grad")

case1 = {
    "params": [
        {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (-1, -1),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        },
        {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (-1, -1),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        },
        {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (-1, -1),
            "ori_format": "ND",
            "range": [(1, 10), (1, 10)]
        },
    ],
    "case_name": "FastGeluGrad_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [{
        "shape": (1, 10),
        "ori_shape": (2, 10),
        "range": ((1, 10), (10, 10)),
        "format": "NHWC",
        "ori_format": "NHWC",
        'dtype': "float16"
    }, {
        "shape": (1, 10),
        "ori_shape": (2, 10),
        "range": ((1, 10), (10, 10)),
        "format": "NHWC",
        "ori_format": "NHWC",
        'dtype': "float16"
    }, {
        "shape": (1, 10),
        "ori_shape": (2, 10),
        "range": ((1, 10), (10, 10)),
        "format": "NHWC",
        "ori_format": "NHWC",
        'dtype': "float16"
    }],
    "case_name": "FastGeluGrad_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
