#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MatrixSetDiagD", None, None)

case1 = {"params": [{"shape": (4, 4, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4, 5),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"},
                    {"shape": (4, 4, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4, 5),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_set_diag_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (3, 7, 5), "dtype": "int8", "format": "NHWC", "ori_shape": (3, 7, 5),"ori_format": "NHWC"},
                    {"shape": (3, 5), "dtype": "int8", "format": "NHWC", "ori_shape": (3, 5),"ori_format": "NHWC"},
                    {"shape": (3, 7, 5), "dtype": "int8", "format": "NHWC", "ori_shape": (3, 7, 5),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "int8", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_set_diag_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (4, 4, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4, 5),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"},
                    {"shape": (4, 3, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 3, 5),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_set_diag_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (4, 4, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4, 5),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_set_diag_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (32,16,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,16,16),"ori_format": "NHWC"},
                    {"shape": (32, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 16),"ori_format": "NHWC"},
                    {"shape": (32,16,16), "dtype": "float16", "format": "NHWC", "ori_shape":(32,16,16),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_set_diag_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)


def calc_expect_func(x1, x2, x3, y):
    input_matrix = x1['value']
    input_diagonal = x2['value']
    input_help = x3['value']

    help_tmp = input_help - 1
    help_y = np.abs(help_tmp)
    res = input_matrix * help_y + input_diagonal * input_help
    return res.astype(y['dtype'])
