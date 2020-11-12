#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("FillD", None, None)

case1 = {"params": [{"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int32"},
                    {"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int32"},
                    (1,)],
         "case_name": "fill_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float16"},
                    {"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float16"},
                    (1,)],
         "case_name": "fill_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"},
                    {"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "float32"},
                    (1,)],
         "case_name": "fill_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int8"},
                    {"shape": (1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int8"},
                    (1,)],
         "case_name": "fill_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

def calc_expect_func(x, y, dims):
    input_Arr = x['value']
    result = np.full(dims, input_Arr).astype(y['dtype'])
    return result

ut_case.add_precision_case("all", {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
                                              (1,)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
                                              (1,)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

