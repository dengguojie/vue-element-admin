#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SplitD", "impl.split_d", "op_select_format")

case1 = {"params": [{"shape": (1024, 1024, 256), "dtype": "uint16", "format": "NCHW", "ori_shape": (1024, 1024, 256),"ori_format": "NCHW"},
                    [], -5, 1],
         "case_name": "split_d_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1024, 1024, 1024), "dtype": "uint16", "format": "NCHW", "ori_shape": (1024, 1024, 1024),"ori_format": "NCHW"},
                    [], 0, 1],
         "case_name": "split_d_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    # ut_case.run()
    ut_case.run("Ascend910")
    exit(0)
