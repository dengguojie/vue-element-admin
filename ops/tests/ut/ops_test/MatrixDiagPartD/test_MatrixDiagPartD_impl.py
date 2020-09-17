#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MatrixDiagPartD", None, None)

case1 = {"params": [{"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_part_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 8192, 8192), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 8192, 8192),"ori_format": "NHWC"},
                    {"shape": (2, 8192, 8192), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 8192, 8192),"ori_format": "NHWC"},
                    {"shape": (2, 8192, 8192), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 8192, 8192),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_part_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2, 4, 3), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 3),"ori_format": "NHWC"},
                    {"shape": (2, 4, 3), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 3),"ori_format": "NHWC"},
                    {"shape": (2, 4, 3), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 3),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_part_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2, 3), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 3),"ori_format": "NHWC"},
                    {"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"},
                    {"shape": (2, 3), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 3),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_part_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2, ), "dtype": "int32", "format": "NHWC", "ori_shape": (2, ),"ori_format": "NHWC"},
                    {"shape": (2, ), "dtype": "int32", "format": "NHWC", "ori_shape": (2, ),"ori_format": "NHWC"},
                    {"shape": (2, ), "dtype": "int32", "format": "NHWC", "ori_shape": (2, ),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_part_d_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)


if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)