"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

HardSwishGrad ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("HardSwishGrad", None, None)


# pylint: disable=unused-argument
def calc_expect_func(input_grad, input_x, output_y):
    dtype = input_grad["dtype"]
    if dtype == "fp16" or dtype == "float16":
        sdtype = np.float16
    elif dtype == "fp32" or dtype == "float32":
        sdtype = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)
    cpu_res = input_x["value"]
    cpu_res = cpu_res / 3 + 0.5
    cpu_res[input_x["value"] < -3] = 0
    cpu_res[input_x["value"] > 3] = 1
    cpu_res = input_grad["value"] * cpu_res
    return cpu_res.astype(sdtype)


case1 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 8, 375), "ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 8, 375), "ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 8, 375), "ori_format": "ND"}],
         "case_name": "hard_swish_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2, 1, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (2, 1, 16), "ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (2, 1, 16), "ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (2, 1, 16), "ori_format": "ND"}],
         "case_name": "hard_swish_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 1), "shape": (16, 1), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 1), "shape": (16, 1), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 1), "shape": (16, 1), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (2, 5, 5), "shape": (2, 5, 5), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (2, 5, 5), "shape": (2, 5, 5), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (2, 5, 5), "shape": (2, 5, 5), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, 2, 4, 3), "shape": (1, 2, 4, 3), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, 2, 4, 3), "shape": (1, 2, 4, 3), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, 2, 4, 3), "shape": (1, 2, 4, 3), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, 2, 4, 7), "shape": (1, 2, 4, 7), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, 2, 4, 7), "shape": (1, 2, 4, 7), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, 2, 4, 7), "shape": (1, 2, 4, 7), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)
})

if __name__ == '__main__':
    ut_case.run()
