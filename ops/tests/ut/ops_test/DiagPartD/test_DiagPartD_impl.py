#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
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

def calc_expect_func(x, assist, y):
    res_vmul = x['value'] * assist['value']
    sum_dims = []
    len_output = len(x['shape']) // 2
    for dims in range(len_output):
        sum_dims.append(dims + len_output)
    has_improve_precision = False
    if x['dtype'] == "int32":
        res_vmul = res_vmul.astype("float32")
        has_improve_precision = True
    res = np.sum(res_vmul, tuple(sum_dims))
    if has_improve_precision:
        res = np.round(res)
    return res

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (2,4,2,4), "dtype": "float16", "format": "ND", "ori_shape": (2,4,2,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2,4,2,4), "dtype": "float16", "format": "ND", "ori_shape": (2,4,2,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2,4), "dtype": "float16", "format": "ND", "ori_shape": (2,4),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (2,3,4,2,3,4), "dtype": "float16", "format": "ND", "ori_shape": (2,3,4,2,3,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2,3,4,2,3,4), "dtype": "float16", "format": "ND", "ori_shape": (2,3,4,2,3,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2,3,4), "dtype": "float16", "format": "ND", "ori_shape": (2,3,4),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (2,3,11,32,2,3,11,32), "dtype": "float16", "format": "ND", "ori_shape": (2,3,11,32,2,3,11,32),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2,3,11,32,2,3,11,32), "dtype": "float16", "format": "ND", "ori_shape": (2,3,11,32,2,3,11,32),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2,3,11,32), "dtype": "float16", "format": "ND", "ori_shape": (2,3,11,32),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                         })
