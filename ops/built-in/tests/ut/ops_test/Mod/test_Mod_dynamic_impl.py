#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT


ut_case = OpUT("Mod", "impl.dynamic.mod", "mod")

case1 = {
    "params": [
        {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"},
        {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"},
        {"shape": (-1,), "ori_shape": (2,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"}
    ],
    "case_name": "mod_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910", "Ascend710", "Ascend610"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910")
