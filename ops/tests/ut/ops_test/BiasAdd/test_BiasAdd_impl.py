"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

BiasAdd ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("BiasAdd", None, None)

case1 = {"params": [{"shape": (2,3,4), "dtype": "float16", "format": "NHWC", "ori_shape": (2,3,4),"ori_format": "NHWC"}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NHWC", "ori_shape": (4,),"ori_format": "NHWC"},
                    {"shape": (2,3,4), "dtype": "float16", "format": "NHWC", "ori_shape":(2,3,4),"ori_format": "NHWC"},
                    ],
         "case_name": "BiasAdd_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2,3,4,5), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,4,5),"ori_format": "NHWC"}, #x
                    {"shape": (5,), "dtype": "float32", "format": "NHWC", "ori_shape": (5,),"ori_format": "NHWC"},
                    {"shape": (2,3,4,5), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,4,5),"ori_format": "NHWC"},
                    ],
         "case_name": "BiasAdd_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2,5,3,4,4), "dtype": "float16", "format": "NCHW", "ori_shape": (2,5,3,4,4),"ori_format": "NCHW"}, #x
                    {"shape": (4, ), "dtype": "float16", "format": "NCHW", "ori_shape": (4,),"ori_format": "NCHW"},
                    {"shape": (2,5,3,4,4), "dtype": "float16", "format": "NCHW", "ori_shape": (2,5,3,4,4),"ori_format": "NCHW"},
                    ],
         "case_name": "BiasAdd_3",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)

def calc_expect_func(x, bias, y):
    res = x['value'] + bias['value']
    return res

precision_case1 = {"params": [{"shape": (10, 10, 10, 20), "dtype": "float16", "format": "ND", "ori_shape": (10, 10, 10, 20),"ori_format": "ND","param_type":"input"},
                              {"shape": (20,), "dtype": "float16", "format": "ND", "ori_shape": (20,),"ori_format": "ND","param_type":"input"},
                              {"shape": (10, 10, 10, 20), "dtype": "float16", "format": "ND", "ori_shape": (10, 10, 10, 20),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (2,3), "dtype": "float16", "format": "ND", "ori_shape": (2,3),"ori_format": "ND","param_type":"input"},
                              {"shape": (3,), "dtype": "float16", "format": "ND", "ori_shape": (3,),"ori_format": "ND","param_type":"input"},
                              {"shape": (2,3), "dtype": "float16", "format": "ND", "ori_shape": (2,3),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (2,3,4,5,6), "dtype": "float32", "format": "ND", "ori_shape": (2,3,4,5,6),"ori_format": "ND","param_type":"input"},
                              {"shape": (6,), "dtype": "float32", "format": "ND", "ori_shape": (6,),"ori_format": "ND","param_type":"input"},
                              {"shape": (2,3,4,5,6), "dtype": "float32", "format": "ND", "ori_shape": (2,3,4,5,6),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910",precision_case1)
ut_case.add_precision_case("Ascend910",precision_case2)
ut_case.add_precision_case("Ascend910",precision_case3)

