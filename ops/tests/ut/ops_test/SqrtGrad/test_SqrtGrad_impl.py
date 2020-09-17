# -*- coding:utf-8 -*-
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

ut_case.add_case(["Ascend910"], case_small_shape_fp32)
ut_case.add_case(["Ascend910"], case_big_shape_prime_fp32)

# ut_case.add_case(["Ascend310"], case1)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
