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

SqrtGrad ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("SqrtGrad")


case_small_shape_fp32 = {
    "params":
        [
            {
                "shape": (2, 3),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (2, 3),
                "ori_format": "ND"
            },
            {
                "shape": (2, 3),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (2, 3),
                "ori_format": "ND"
            },
            {
                "shape": (2, 3),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (2, 3),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_sqrt_grad_small_shape_fp32',
    "expect": "success"
}

case_big_shape_prime_fp32 = {
    "params":
        [
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_sqrt_grad_big_shape_prime_fp32',
    "expect": "success"
}

case_different_shape_prime_fp32 = {
    "params":
        [
            {
                "shape": (9973, 1),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 1),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_sqrt_grad_different_shape_prime_fp32',
    "expect": RuntimeError
}

ut_case.add_case(["Ascend910"], case_small_shape_fp32)
ut_case.add_case(["Ascend910"], case_big_shape_prime_fp32)
ut_case.add_case(["Ascend910"], case_different_shape_prime_fp32)

# ut_case.add_case(["Ascend310"], case1)


def calc_expect_func(x, dx, out):
    dtype = x["dtype"]
    typeMap = {
        'float16': np.float16,
        'float32': np.float32,
        'int32': np.int32,
        'int8': np.int8,
        'uint8': np.uint8
    }
    sdtype = typeMap[dtype]

    input_A_Arr = x["value"]
    input_B_Arr = dx["value"]

    if dtype=="float16":
        input_A_Arr.astype(np.float32)
        input_B_Arr.astype(np.float32)

    outputArr=np.divide(input_B_Arr, np.multiply(input_A_Arr, 2))
    if dtype=="float16":
        outputArr.astype(np.float16)

    return outputArr.astype(sdtype)

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 64), "shape": (2, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 64), "shape": (2, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 64), "shape": (2, 64), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (73, 64), "shape": (73, 64), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (73, 64), "shape": (73, 64), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (73, 64), "shape": (73, 64), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.002, 0.002)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (15, 3, 7, 5, 3, 2), "shape": (15, 3, 7, 5, 3, 2), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (15, 3, 7, 5, 3, 2), "shape": (15, 3, 7, 5, 3, 2), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (15, 3, 7, 5, 3, 2), "shape": (15, 3, 7, 5, 3, 2), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.002, 0.002)
})
if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
