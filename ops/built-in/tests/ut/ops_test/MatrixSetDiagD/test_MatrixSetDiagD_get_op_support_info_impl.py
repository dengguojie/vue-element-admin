#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MatrixSetDiagD", "matrix_set_diag_d", "get_op_support_info")

case1 = {"params": [{"shape": (4, 4, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4, 5),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"},
                    {"shape": (4, 4, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4, 5),"ori_format": "NHWC"},
                    {"shape": (4, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_set_diag_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

