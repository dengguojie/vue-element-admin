#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("SoftmaxV2", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "softmax_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    [0,1]],
         "case_name": "softmax_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "softmax_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "softmax_v2_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    [0,1]],
         "case_name": "softmax_v2_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [0]],
         "case_name": "softmax_v2_6hd_n",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [-1]],
         "case_name": "softmax_v2_6hd_c",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [0]],
         "case_name": "softmax_v2_6hd_n",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [-1]],
         "case_name": "softmax_v2_6hd_c",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (16, 16, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 16, 16),"ori_format": "ND"},
                     {"shape": (16, 16, 1, 1, 16 ,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 16, 16),"ori_format": "ND"},
                     [-1]],
         "case_name": "softmax_v2_nz",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [{"shape": (16, 16, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 50, 50),"ori_format": "ND"},
                     {"shape": (16, 16, 4, 4, 16 ,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 50, 50),"ori_format": "ND"},
                     [-1]],
         "case_name": "softmax_v2_nz_01",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case12 = {"params": [{"shape": (16, 1, 4, 4, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16, 4, 4, 4),"ori_format": "NHWC"},
                     {"shape": (16, 1, 4, 4, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16, 4, 4, 4),"ori_format": "NHWC"},
                     [-1]],
         "case_name": "softmax_v2_5hd_01",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case13 = {"params": [{"shape": (8, 6, 546, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 8732, 81),"ori_format": "ND"},
                     {"shape": (8, 6, 546, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 8732, 81),"ori_format": "ND"},
                     [2]],
          "case_name": "softmax_v2_nz_13",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)

# precision cases
## need axis is list

def calc_expect_func(x, y, axis):
    input_Arr = x['value']
    data_max = np.max(input_Arr, axis, keepdims=True).astype(np.float16)

    data_sub = np.subtract(input_Arr, data_max).astype(np.float32)
    expres = np.exp(data_sub).astype(np.float32)
    sumre = np.sum(expres, axis, keepdims=True).astype(np.float32)
    result = (expres / sumre).astype(y['dtype'])
    return result

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "output"},
                                                    1],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND", "param_type": "output"},
                                                    0],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (32,2,4,16), "dtype": "float32", "format": "ND", "ori_shape": (32,2,4,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32,2,4,16), "dtype": "float32", "format": "ND", "ori_shape": (32,2,4,16),"ori_format": "ND", "param_type": "output"},
                                                    1],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "output"},
                                                    -1],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })


