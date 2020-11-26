#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Pooling", None, None)

case1 = {"params": [{"shape": (1,2,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,4,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (1,2,4,16,16),"ori_format": "FRACTAL_Z"},
                    {"shape": (1,2,4,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,2,4,16,16),"ori_format": "NCHW"},
                    {"shape": (1,2,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "pooling_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,4,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,4,16,16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1,2,4,16,16),"ori_format": "FRACTAL_Z"},
                    {"shape": (1,2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,2,4,16,16),"ori_format": "NCHW"},
                    {"shape": (1,2,4,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "pooling_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"}],
         "case_name": "pooling_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"}],
         "case_name": "pooling_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "pooling_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
