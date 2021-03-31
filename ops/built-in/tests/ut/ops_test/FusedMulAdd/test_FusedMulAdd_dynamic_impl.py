#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("FusedMulAdd", "impl.dynamic.fused_mul_add", "fused_mul_add")

case1 = {
    "params": [
        {
            "shape": (-1, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        },
        {
            "shape": (2, 2),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(2, 2), (2, 2)]
        },
        {
            "shape": (-1, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        },
        {
            "shape": (-1, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        }
    ],
    "case_name": "FusedMulAdd_1",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run(["Ascend910A"])
