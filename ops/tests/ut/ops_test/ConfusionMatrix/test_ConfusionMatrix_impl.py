#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ConfusionMatrix", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "int16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)