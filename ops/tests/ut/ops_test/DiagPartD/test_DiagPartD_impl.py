#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("DiagPartD", None, None)

case1 = {"params": [{"shape": (2,4,2,4), "dtype": "int32", "format": "NCHW", "ori_shape": (2,4,2,4),"ori_format": "NCHW"},
                    {"shape": (2,4,2,4), "dtype": "int32", "format": "NCHW", "ori_shape": (2,4,2,4),"ori_format": "NCHW"},
                    {"shape": (2,4), "dtype": "int32", "format": "NCHW", "ori_shape": (2,4),"ori_format": "NCHW"}],
         "case_name": "diag_part_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2,3,4,2,3,4), "dtype": "float16", "format": "NCHW", "ori_shape": (2,3,4,2,3,4),"ori_format": "NCHW"},
                    {"shape": (2,3,4,2,3,4), "dtype": "float16", "format": "NCHW", "ori_shape": (2,3,4,2,3,4),"ori_format": "NCHW"},
                    {"shape": (2,3,4), "dtype": "float16", "format": "NCHW", "ori_shape": (2,3,4),"ori_format": "NCHW"}],
         "case_name": "diag_part_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (2,3,11,32,2,3,11,32), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,11,32,2,3,11,32),"ori_format": "NCHW"},
                    {"shape": (2,3,11,32,2,3,11,32), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,11,32,2,3,11,32),"ori_format": "NCHW"},
                    {"shape": (2,3,11,32), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,11,32),"ori_format": "NCHW"}],
         "case_name": "diag_part_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910")
