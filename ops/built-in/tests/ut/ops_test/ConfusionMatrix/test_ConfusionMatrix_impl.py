#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import tensorflow as tf
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
case5 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    None,
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (2,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    None,
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_5",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (2,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_6",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_7",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_8",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_9",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "float16"],
         "case_name": "confusion_matrix_10",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case11 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_11",
         "expect": "failed",
         "format_expect": [],
         "support_expect": True}
case12 = {"params": [{"shape": (1,3,4,8), "dtype": "float16", "format": "NCHW", "ori_shape": (1,3,4,8),"ori_format": "NCHW"},
                    {"shape": (1,3,4,8), "dtype": "float16", "format": "NCHW", "ori_shape": (1,3,4,8),"ori_format": "NCHW"},
                    {"shape": (1,3,4,8), "dtype": "float32", "format": "NCHW", "ori_shape": (1,3,4,8),"ori_format": "NCHW"},
                    {"shape": (1,3,4,8), "dtype": "int8", "format": "NCHW", "ori_shape": (1,3,4,8),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_12",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}    #line 369  confusion_matrix_ir start
case13 = {"params": [{"shape": (1,3,4,8), "dtype": "float16", "format": "NCHW", "ori_shape": (1,3,4,8),"ori_format": "NCHW"},
                    {"shape": (1,3,4,8), "dtype": "float16", "format": "NCHW", "ori_shape": (1,3,4,8),"ori_format": "NCHW"},
                    {"shape": (1,3,4,8), "dtype": "float16", "format": "NCHW", "ori_shape": (1,3,4,8),"ori_format": "NCHW"},
                    {"shape": (1,3,4,8), "dtype": "int8", "format": "NCHW", "ori_shape": (1,3,4,8),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_13",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case14 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_14",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case15 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_15",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case16 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case16 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_16",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case17 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_17",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case18 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_18",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case19 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_19",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case20 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_20",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case21 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_21",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case22 = {"params": [{"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_22",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case23 = {"params": [{"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_23",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case24 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_24",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case25 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_25",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case26 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_26",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case27 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_27",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case28 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_28",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case29 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_29",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case30 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_30",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case31 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_31",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case32 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_32",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case33 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_33",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case34 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_34",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case35 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_35",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case36 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int8", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_36",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case37 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_37",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case38 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    2, "int32"],
         "case_name": "confusion_matrix_38",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}








ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case14)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case15)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case16)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case17)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case18)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case19)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case20)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case21)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case22)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case23)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case24)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case25)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case26)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case27)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case28)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case29)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case30)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case31)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case32)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case33)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case34)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case35)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case36)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case37)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case38)



def test_op_get_op_support_info(test_arg): 
    from impl.confusion_matrix import get_op_support_info
    get_op_support_info({"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                        {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                        {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                        {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                        2, "float16"
                        "get_op_support_info_case1")  
ut_case.add_cust_test_func(test_func=test_op_get_op_support_info)

def calc_expect_func(labels, predictions, weights, y, num_classes, dtype):
    out = tf.math.confusion_matrix(labels['value'], predictions['value'], 
                                   num_classes=num_classes, weights=weights['value'], dtype=dtype)
    with tf.Session() as sess:
        res = sess.run(out)
    return res

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2,2), "dtype": "float16", "format": "ND", "ori_shape": (2,2),"ori_format": "ND", "param_type": "output"},
                                                    2, "float16"],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                         })