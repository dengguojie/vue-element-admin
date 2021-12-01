
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("DecodeWheelsTarget", None, None)

case1 = {"params": [{"shape": (1,8), "dtype": "float16", "format": "ND", "ori_shape": (1,8),"ori_format": "ND"},
                    {"shape": (1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,4),"ori_format": "ND"},
                    {"shape": (1,8), "dtype": "float16", "format": "ND", "ori_shape": (1,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,8), "dtype": "float16", "format": "ND", "ori_shape": (2,8),"ori_format": "ND"},
                    {"shape": (2,4), "dtype": "float16", "format": "ND", "ori_shape": (2,4),"ori_format": "ND"},
                    {"shape": (2,8), "dtype": "float16", "format": "ND", "ori_shape": (2,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (7,8), "dtype": "float16", "format": "ND", "ori_shape": (7,8),"ori_format": "ND"},
                    {"shape": (7,4), "dtype": "float16", "format": "ND", "ori_shape": (7,4),"ori_format": "ND"},
                    {"shape": (7,8), "dtype": "float16", "format": "ND", "ori_shape": (7,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (16,8), "dtype": "float16", "format": "ND", "ori_shape": (16,8),"ori_format": "ND"},
                    {"shape": (16,4), "dtype": "float16", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"},
                    {"shape": (16,8), "dtype": "float16", "format": "ND", "ori_shape": (16,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (777,8), "dtype": "float16", "format": "ND", "ori_shape": (777,8),"ori_format": "ND"},
                    {"shape": (777,4), "dtype": "float16", "format": "ND", "ori_shape": (777,4),"ori_format": "ND"},
                    {"shape": (777,8), "dtype": "float16", "format": "ND", "ori_shape": (777,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

err1 = {"params": [{"shape": (100,8), "dtype": "float32", "format": "ND", "ori_shape": (100,8),"ori_format": "ND"},
                   {"shape": (100,4), "dtype": "float32", "format": "ND", "ori_shape": (100,4),"ori_format": "ND"},
                   {"shape": (100,8), "dtype": "float32", "format": "ND", "ori_shape": (100,8),"ori_format": "ND"}],
        "case_name": "err_1",
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True}
err2 = {"params": [{"shape": (800,), "dtype": "float16", "format": "ND", "ori_shape": (800,),"ori_format": "ND"},
                   {"shape": (400,), "dtype": "float16", "format": "ND", "ori_shape": (400,),"ori_format": "ND"},
                   {"shape": (800,), "dtype": "float16", "format": "ND", "ori_shape": (800,),"ori_format": "ND"}],
        "case_name": "err_2",
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True}
err3 = {"params": [{"shape": (100, 6), "dtype": "float16", "format": "ND", "ori_shape": (100, 6),"ori_format": "ND"},
                   {"shape": (100, 4), "dtype": "float16", "format": "ND", "ori_shape": (100, 4),"ori_format": "ND"},
                   {"shape": (100, 6), "dtype": "float16", "format": "ND", "ori_shape": (100, 6),"ori_format": "ND"}],
        "case_name": "err_3",
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True}

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)
ut_case.add_case("Ascend910A", case5)
ut_case.add_case("Ascend910A", err1)
ut_case.add_case("Ascend910A", err2)
ut_case.add_case("Ascend910A", err3)

def calc_expect_func(inputA, inputB, output):
    input_shape_x = inputA['shape']
    data_a = inputA['value']
    data_b = inputB['value']
    a_x1y1x2y2x3y3x4y4 = np.reshape(data_a, (-1, 4, 2))
    a_x1y1, a_x2y2, a_x3y3, a_x4y4 = np.split(a_x1y1x2y2x3y3x4y4, 4, axis=1)
    b_x1y1x2y2 = np.reshape(data_b, (-1, 2, 2))
    b_x1y1, b_x2y2 = np.split(b_x1y1x2y2, 2, axis=1)
    waha = b_x2y2 - b_x1y1
    xaya = (b_x2y2 + b_x1y1) * 0.5
    c_x1y1 = a_x1y1 * waha + xaya
    c_x2y2 = a_x2y2 * waha + xaya
    c_x3y3 = a_x3y3 * waha + xaya
    c_x4y4 = a_x4y4 * waha + xaya
    c = np.concatenate([c_x1y1, c_x2y2, c_x3y3, c_x4y4], axis=1)
    data_c = np.reshape(c, input_shape_x)
    return data_c

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 8), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 8), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (8, 8), "dtype": "float16", "format": "ND", "ori_shape": (8, 8),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (8, 4), "dtype": "float16", "format": "ND", "ori_shape": (8, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (8, 8), "dtype": "float16", "format": "ND", "ori_shape": (8, 8),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16, 8), "dtype": "float16", "format": "ND", "ori_shape": (16, 8),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 4), "dtype": "float16", "format": "ND", "ori_shape": (16, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 8), "dtype": "float16", "format": "ND", "ori_shape": (16, 8),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })


