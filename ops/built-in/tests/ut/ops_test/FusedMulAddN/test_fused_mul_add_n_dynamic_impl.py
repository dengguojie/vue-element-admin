#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
test for FusedMulAddN
'''
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("FusedMulAddN", "impl.dynamic.fused_mul_add_n", "fused_mul_add_n")

case1 = {
    "params": [
        {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1, 1),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1, 1),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100)]
        },
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "range": [(1, None)]
        },
        {
            "shape": (-1, -1),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1, 1),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100)]
        },
    ],
    "case_name": "FusedMulAddN_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [{
        "shape": (-1, 10),
        "ori_shape": (2, 10),
        "range": ((1, None), (10, 10)),
        "format": "NHWC",
        "ori_format": "NHWC",
        'dtype': "float16"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "range": ((1, 1),),
        "format": "NHWC",
        "ori_format": "NHWC",
        'dtype': "float32"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "range": ((1, 1),),
        "format": "NHWC",
        "ori_format": "NHWC",
        'dtype': "float32"
    }, {
        "shape": (-1, 10),
        "ori_shape": (2, 10),
        "range": ((1, None), (10, 10)),
        "format": "NHWC",
        "ori_format": "NHWC",
        'dtype': "float16"
    }],
    "case_name": "FusedMulAddN_2",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
