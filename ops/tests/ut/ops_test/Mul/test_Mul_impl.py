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

Mul ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("Mul", None, None)

case1 = {"params": [{"shape": (8192, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (8192, 1),"ori_format": "NHWC"},
                    {"shape": (8192, 100), "dtype": "float32", "format": "NHWC", "ori_shape": (8192, 100),"ori_format": "NHWC"},
                    {"shape": (8192, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (8192, 1),"ori_format": "NHWC"}],
         "case_name": "mul_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (10241,), "dtype": "float16", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "float16", "format": "NHWC", "ori_shape": (10, 10241),"ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "float16", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"}
                    ],
         "case_name": "mul_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (10241,), "dtype": "float32", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "float32", "format": "NHWC", "ori_shape": (10, 10241),"ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "float32", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"}
                    ],
         "case_name": "mul_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (10241,), "dtype": "int8", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "int8", "format": "NHWC", "ori_shape": (10, 10241),"ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "int8", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"}
                    ],
         "case_name": "mul_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

def calc_expect_func(input_a, input_b, output_y):
    return np.multiply(input_a["value"], input_b["value"]).astype(input_a["dtype"])

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (92, 1), "shape": (92, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (92, 100), "shape": (92, 100), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (92, 100), "shape": (92, 100), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    ut_case.run()
    exit(0)