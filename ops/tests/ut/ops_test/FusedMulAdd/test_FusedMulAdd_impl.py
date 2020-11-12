#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("FusedMulAdd", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

def calc_expect_func(x1, x2, x3, y):
    mul = x1['value'] * x2['value']
    res = mul + x3['value']
    res = res.astype(y['dtype'])
    return res

precision_case1 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "param_type":"input"},
                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case2 = {"params": [{"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ), "param_type":"input"},
                              {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case3 = {"params": [{"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32), "param_type":"input"},
                              {"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case4 = {"params": [{"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16), "param_type":"input"},
                              {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case1)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case2)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case3)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case4)

