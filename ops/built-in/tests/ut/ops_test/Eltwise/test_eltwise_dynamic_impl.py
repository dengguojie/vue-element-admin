#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
eltwise ut case
"""
import tbe
from op_test_frame.ut import OpUT
ut_case = OpUT("Eltwise", "impl.dynamic.eltwise", "eltwise")

case1 = {"params": [[{"shape": (-1, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                     {"shape": (-1, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]}],
                    {"shape": (-1, 16), "dtype": "float16",
                     "format": "NCHW", "ori_shape": (16, 16),
                     "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                    1],
         "case_name": "eltwise_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
