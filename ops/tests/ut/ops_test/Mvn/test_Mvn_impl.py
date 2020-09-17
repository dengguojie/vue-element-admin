#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("Mvn", None, None)

case1 = {"params": [{"shape": (2, 3, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"},
                    {"shape": (2, 3, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"}],
         "case_name": "mvn_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2, 3, 2, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"},
                    {"shape": (2, 3, 2, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"}],
         "case_name": "mvn_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run()
    exit(0)