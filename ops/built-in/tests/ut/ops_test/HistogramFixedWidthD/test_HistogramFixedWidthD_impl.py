#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("HistogramFixedWidthD", None, None)

case1 = {"params": [{"shape": (11, 33), "dtype": "float16", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (11, 33), "dtype": "float16", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND"},
                    1000000],
         "case_name": "histogram_fixed_width_d_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1024, 1024, 256), "dtype": "float16", "format": "ND", "ori_shape": (1024, 1024, 256),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (1024, 1024, 256), "dtype": "float16", "format": "ND", "ori_shape": (1024, 1024, 256),"ori_format": "ND"},
                    10],
         "case_name": "histogram_fixed_width_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)