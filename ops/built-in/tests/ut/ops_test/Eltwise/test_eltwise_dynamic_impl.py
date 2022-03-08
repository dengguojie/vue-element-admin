#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
eltwise ut case
"""
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
                    0],
         "case_name": "eltwise_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [[{"shape": (-1, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                     {"shape": (-1, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]}],
                    {"shape": (-1, 16), "dtype": "float16",
                     "format": "NCHW", "ori_shape": (16, 16),
                     "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                    1],
         "case_name": "eltwise_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [[{"shape": (-1, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                     {"shape": (-1, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]}],
                    {"shape": (-1, 16), "dtype": "float16",
                     "format": "NCHW", "ori_shape": (16, 16),
                     "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                    2],
         "case_name": "eltwise_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
