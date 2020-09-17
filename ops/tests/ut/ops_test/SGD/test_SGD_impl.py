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

SGD ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("SGD",
               "impl.sgd",
               "sgd")

case_small_shape_scalar_fp32 = {
    "params":
        [
            {
                "shape": (1, ),  # parameters
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # gradient
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # learning_rate
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1,),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),   # momentum
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),  # stat
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # parameters
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            # dampening,weight_decay,nesterov
            0.0,
            0.0,
            False
        ],
    "case_name": 'test_sgd_small_shape_scalar_fp32',
    "expect": "success"
}

case_medium_shape_fp32 = {
    "params":
        [
            {
                "shape": (9973, 13, 8297),  # parameters
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8297),  # gradient
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # learning_rate
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 13, 8297),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),   # momentum
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 13, 8297),  # stat
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8297),  # parameters
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "NC1HWC0"
            },
            # dampening,weight_decay,nesterov
            0.0,
            0.0,
            False
        ],
    "case_name": 'test_sgd_small_shape_fp32',
    "expect": "success"
}

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_scalar_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_medium_shape_fp32)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
