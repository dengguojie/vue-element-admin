#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Mask2Argmax", None, None)


case1 = {"params": [{"shape": (2,2,96,144,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,96,144,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,48,72,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,48,72,16),"ori_format": "NC1HWC0"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (13888,),"ori_format": "NC1HWC0"},
                    [1, 2, 2, 1],
                    [1, 2, 2, 1],
                    "SAME",
                    [1, 1, 1, 1]],
         "case_name": "mask2_argmax_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (32,4,112,112,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,4,112,112,16),"ori_format": "NHWC"},
                    {"shape": (32,4,56,56,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,4,56,56,16),"ori_format": "NHWC"},
                    {"shape": (32,4,9,197,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (32,4,9,197,16),"ori_format": "NHWC"},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    "SAME",
                    [1, 1, 1, 1]],
         "case_name": "mask2_argmax_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)

if __name__ == '__main__':
    # ut_case.run()
    ut_case.run("Ascend310")
    exit(0)