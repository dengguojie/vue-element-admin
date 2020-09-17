#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("Crop", None, None)

case1 = {"params": [{"shape": (2, 3, 2, 16,6,7), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 3, 2, 16,6,7),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1, 8,5,6), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1, 8,5,6),"ori_format": "NCHW"},
                    {"shape": (2, 2, 1, 8,5,6), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1, 8,5,6),"ori_format": "NCHW"},
                    0,  [0, 0, 0, 0, 0, 0]],
         "case_name": "crop_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 3, 2, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"},
                    {"shape": (2,3,2,3), "dtype": "int32", "format": "NCHW", "ori_shape": (2,3,2,3),"ori_format": "NCHW"},
                    {"shape": (2,3,2,3), "dtype": "int32", "format": "NCHW", "ori_shape": (2,3,2,3),"ori_format": "NCHW"},
                    1,  [5]],
         "case_name": "crop_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")
