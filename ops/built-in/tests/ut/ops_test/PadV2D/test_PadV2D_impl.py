#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("PadV2D", None, None)

case1 = {"params": [{"shape": (32, 128, 1024), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 128, 1024),"ori_format": "NCHW"},
                    {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,),"ori_format": "NCHW"},
                    {"shape": (32, 128, 1024), "dtype": "float16", "format": "NCHW", "ori_shape": (32, 128, 1024),"ori_format": "NCHW"},
                    [[0, 0],[0, 384],[0, 0]]],
         "case_name": "pad_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 2, 1024*240), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1024*240),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1024*240), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1024*240),"ori_format": "NCHW"},
                    [[0, 0],[7, 7],[0, 7]]],
         "case_name": "pad_v2_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,),"ori_format": "NCHW"},
                    [[0,3]]],
         "case_name": "pad_v2_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,2,9), "dtype": "float32", "format": "NCHW", "ori_shape": (2,2,9),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2,2,9), "dtype": "float32", "format": "NCHW", "ori_shape": (2,2,9),"ori_format": "NCHW"},
                    [[0, 0],[9, 7],[0, 0]]],
         "case_name": "pad_v2_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0],[0, 7],[0, 7]]],
         "case_name": "pad_v2_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0],[0, 16],[0, 0]]],
         "case_name": "pad_v2_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float16", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0],[0, 16],[0, 0]]],
         "case_name": "pad_v2_d_7",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (2, 2, 1027), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 1027),"ori_format": "NHWC"},
                    {"shape": (8,), "dtype": "float16", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0],[0, 16],[0, 0]]],
         "case_name": "pad_v2_d_8",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0, 0],[0, 16],[0, 0]]],
         "case_name": "pad_v2_d_6",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (2, 2, 1027), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "int8", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0],[0, 16],[1, 1]]],
         "case_name": "pad_v2_d_10",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case11 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[1, 0],[0, 16],[1, 1]]],
         "case_name": "pad_v2_d_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case12 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 1],[0, 16],[1, 1]]],
         "case_name": "pad_v2_d_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case13 = {"params": [{"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1027), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 1027),"ori_format": "NCHW"},
                    [[0, 0],[0, 0],[1, 1]]],
         "case_name": "pad_v2_d_13",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)
if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)

