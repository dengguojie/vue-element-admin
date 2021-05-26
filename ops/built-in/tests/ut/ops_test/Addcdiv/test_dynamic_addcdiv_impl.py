#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Addcdiv", "impl.dynamic.addcdiv", "addcdiv")

case1 = {
    "params": [
        {"shape": (-1, -1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 16, 16, 1), "ori_format": "ND", "range": [(1, 3), (1, 16), (16, 16), (1, 1)]},
        {"shape": (-1, ), "dtype": "float16", "format": "ND", "ori_shape": (16, ), "ori_format": "ND", "range": [(1, 16), ]},
        {"shape": (-1, ), "dtype": "float16", "format": "ND", "ori_shape": (16, ), "ori_format": "ND", "range": [(1, 16), ]},
        {"shape": (-1, ), "dtype": "float16", "format": "ND", "ori_shape": (16, ), "ori_format": "ND", "range": [(1, 16), ]},
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 16, 16, 16), "ori_format": "ND", "range": [(1, 3), (1, 16), (16, 16), (1, 16) ]}
    ],
    "case_name": "addcdiv_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
