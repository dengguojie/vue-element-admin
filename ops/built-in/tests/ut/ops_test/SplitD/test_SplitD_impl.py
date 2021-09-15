#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SplitD", None, None)

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

case3 = {"params": [{"shape": (2, 1, 16, 16), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (2, 1, 16, 16),"ori_format": "NCHW"},
                    [{"shape": (1, 1, 16, 16), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (1, 1, 16, 16),"ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 16), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (1, 1, 16, 16),"ori_format": "NCHW"}], 0, 2],
         "case_name": "split_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (48000, 256), "dtype": "float16", "format": "ND",
                     "ori_shape": (48000, 256),"ori_format": "ND"},
                    [{"shape": (48000, 64), "dtype": "float16", "format": "ND",
                     "ori_shape": (48000, 64),"ori_format": "ND"},
                     {"shape": (48000, 64), "dtype": "float16", "format": "ND",
                     "ori_shape": (48000, 64),"ori_format": "ND"},
                     {"shape": (48000, 64), "dtype": "float16", "format": "ND",
                     "ori_shape": (48000, 64),"ori_format": "ND"},
                     {"shape": (48000, 64), "dtype": "float16", "format": "ND",
                     "ori_shape": (48000, 64),"ori_format": "ND"},], -1, 4],
         "case_name": "split_d_v_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (16, 52, 52, 3, 86), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 52, 52, 3, 86),"ori_format": "ND"},
                     [{"shape": (16, 52, 52, 3, 43), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 52, 52, 3, 43),"ori_format": "ND"},
                     {"shape": (16, 52, 52, 3, 43), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 52, 52, 3, 43),"ori_format": "ND"},],-1, 2],
         "case_name": "split_d_v_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (8, 46, 46, 63), "dtype": "float16", "format": "ND",
                     "ori_shape": (8, 46, 46, 63),"ori_format": "ND"},
                     [{"shape": (8, 46, 46, 21), "dtype": "float16", "format": "ND",
                     "ori_shape": (8, 46, 46, 21),"ori_format": "ND"},
                     {"shape": (8, 46, 46, 21), "dtype": "float16", "format": "ND",
                     "ori_shape": (8, 46, 46, 21),"ori_format": "ND"},
                     {"shape": (8, 46, 46, 21), "dtype": "float16", "format": "ND",
                     "ori_shape": (8, 46, 46, 21),"ori_format": "ND"},],3, 3],
         "case_name": "split_d_v_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
