#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Upsample", None, None)

case1 = {"params": [{"shape": (1,16,13,13,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,13,13,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,16,13,13,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,13,13,16),"ori_format": "NC1HWC0"},
                    2.3, 4, 2],
         "case_name": "upsample_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,32,16,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,32,16,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,32,16,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,32,16,16,16),"ori_format": "NC1HWC0"},
                    1.0, 2, 1],
         "case_name": "upsample_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,16,1,2048,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,1,2048,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,16,1,2048,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,1,2048,16),"ori_format": "NC1HWC0"},
                    1.0, 2, 1],
         "case_name": "upsample_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,16,16,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,16,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,16,16,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,16,16,16,16),"ori_format": "NC1HWC0"},
                    1.0, 20, 26],
         "case_name": "upsample_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1,1,1,128,3), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,1,1,128,3),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,128,3), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,1,1,128,3),"ori_format": "NC1HWC0"},
                    1.0, 2, 1],
         "case_name": "upsample_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
