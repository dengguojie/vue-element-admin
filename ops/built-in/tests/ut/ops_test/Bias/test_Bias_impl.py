#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("Bias", None, None)

case1 = {"params": [{"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3,), "dtype": "float16", "format": "NCHW", "ori_shape": (3,),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    1, 1],
         "case_name": "bias_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3, 4), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 4),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    1, 1],
         "case_name": "bias_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (3, 3, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3,), "dtype": "float32", "format": "NCHW", "ori_shape": (3,),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    1, 1],
         "case_name": "bias_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Hi3796CV300CS"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
