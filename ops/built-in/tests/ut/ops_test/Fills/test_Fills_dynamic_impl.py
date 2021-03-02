#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Fills UT test
'''
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("Fills", "impl.dynamic.fills", "fills")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, 1.0],
    "case_name": "Fills_1",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case("Ascend910A", case1)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
