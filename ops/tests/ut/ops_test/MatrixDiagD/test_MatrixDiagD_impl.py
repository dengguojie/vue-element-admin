#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MatrixDiagD", None, None)

case1 = {"params": [{"shape": (2, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 8192), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 8192),"ori_format": "NHWC"},
                    {"shape": (2, 8192, 8192), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 8192, 8192),"ori_format": "NHWC"},
                    {"shape": (2, 8192), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 8192),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4),"ori_format": "NHWC"},
                    {"shape": (2, 4, 3), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4, 3),"ori_format": "NHWC"},
                    {"shape": (2, 4), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 4),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2, ), "dtype": "int32", "format": "NHWC", "ori_shape": (2, ),"ori_format": "NHWC"},
                    {"shape": (2, ), "dtype": "int32", "format": "NHWC", "ori_shape": (2, ),"ori_format": "NHWC"},
                    {"shape": (2, ), "dtype": "int32", "format": "NHWC", "ori_shape": (2, ),"ori_format": "NHWC"}],
         "case_name": "matrix_diag_d_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
         
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

def calc_expect_func(x1, x2, y):
    input_diagonal_dtype = x1['dtype']
    if input_diagonal_dtype in ("int8", "uint8"):
        input_diagonal = x1['value'].astype("float16")
    else:
        input_diagonal = x1['value']
    res = input_diagonal * x2['value']
    res = res.astype(y['dtype'])
    return res

# ut_case.add_precision_case("Ascend910", {"params": [{"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "output"}],
#                                          "calc_expect_func": calc_expect_func,
#                                          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                          })
# ut_case.add_precision_case("Ascend910", {"params": [{"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (100,100), "dtype": "int8", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "output"}],
#                                          "calc_expect_func": calc_expect_func,
#                                          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                          })
# ut_case.add_precision_case("Ascend910", {"params": [{"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (32,128), "dtype": "int8", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "output"}],
#                                          "calc_expect_func": calc_expect_func,
#                                          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                          })
# ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (1,16,512), "dtype": "int8", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "output"}],
#                                          "calc_expect_func": calc_expect_func,
#                                          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                          })
# ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "input"},
#                                                     {"shape": (1024,16), "dtype": "int8", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "output"}],
#                                          "calc_expect_func": calc_expect_func,
#                                          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                          })