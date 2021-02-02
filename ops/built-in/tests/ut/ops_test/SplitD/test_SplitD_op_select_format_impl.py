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

case3 = {"params": [{"shape": (1, 1024, 1024, 1024), "dtype": "uint16", "format": "NCHW", "ori_shape": (1, 1024, 1024, 1024),"ori_format": "NCHW"},
                    [], 1, 1],
         "case_name": "split_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1, 1023, 1024, 1024), "dtype": "uint16", "format": "NCHW", "ori_shape": (1, 1023, 1024, 1024),"ori_format": "NCHW"},
                    [], 1, 1],
         "case_name": "split_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (1, 1024, 1024, 1024), "dtype": "uint16", "format": "NHWC", "ori_shape": (1, 1024, 1024, 1024),"ori_format": "NHWC"},
                    [], 3, 1],
         "case_name": "split_d_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (1, 1024, 1024, 1023), "dtype": "uint16", "format": "NHWC", "ori_shape": (1, 1024, 1024, 1023),"ori_format": "NHWC"},
                    [], 3, 1],
         "case_name": "split_d_6",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (1, 1, 1024, 1024, 1024), "dtype": "uint16", "format": "NDCHW", "ori_shape": (1, 1, 1024, 1024, 1024),"ori_format": "NDCHW"},
                    [], 2, 1],
         "case_name": "split_d_7",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (1, 1, 1023, 1024, 1024), "dtype": "uint16", "format": "NDCHW", "ori_shape": (1, 1, 1023, 1024, 1024),"ori_format": "NDCHW"},
                    [], 2, 1],
         "case_name": "split_d_8",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [{"shape": (1, 1, 1024, 1024, 1024), "dtype": "uint16", "format": "NDHWC", "ori_shape": (1, 1, 1024, 1024, 1024),"ori_format": "NDHWC"},
                    [], 4, 1],
         "case_name": "split_d_9",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (1, 1, 1024, 1024, 1023), "dtype": "uint16", "format": "NDHWC", "ori_shape": (1, 1, 1024, 1024, 1023),"ori_format": "NDHWC"},
                    [], 4, 1],
         "case_name": "split_d_10",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}



ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case10)

if __name__ == '__main__':
    # ut_case.run()
    ut_case.run("Ascend910")
    exit(0)
