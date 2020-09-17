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

DepthwiseWeight6DTo4D ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("DepthwiseWeight6DTo4D",
               "impl.depthwise_weight_6d_2_4d",
               "depthwise_weight_6d_2_4d")

case_small_shape_fp16 = {
    "params":
        [
            {
                "shape": (1, 3, 5, 1, 16, 16),
                "format": "C1HWNCoC0",
                "dtype": "float16",
                "ori_shape": (1, 3, 5, 1, 16, 16),
                "ori_format": "C1HWNCoC0"
            },
            {
                "shape": (3, 5, 15, 1),
                "format": "HWCN",
                "dtype": "float16",
                "ori_shape": (3, 5, 15, 1),
                "ori_format": "HWCN"
            },
            "C1HWNCoC0",
            "HWCN"
        ],
    "case_name": 'test_depthwise6d24d_small_shape_fp16',
    "expect": "success"
}

case_small_shape_fp32 = {
    "params":
        [
            {
                "shape": (1, 3, 5, 1, 16, 16),
                "format": "C1HWNCoC0",
                "dtype": "float32",
                "ori_shape": (1, 3, 5, 1, 16, 16),
                "ori_format": "C1HWNCoC0"
            },
            {
                "shape": (3, 5, 15, 1),
                "format": "HWCN",
                "dtype": "float32",
                "ori_shape": (3, 5, 15, 1),
                "ori_format": "HWCN"
            },
            "C1HWNCoC0",
            "HWCN"
        ],
    "case_name": 'test_depthwise6d24d_small_shape_fp32',
    "expect": "success"
}

case_small_shape_int32 = {
    "params":
        [
            {
                "shape": (1, 3, 5, 1, 16, 16),
                "format": "C1HWNCoC0",
                "dtype": "int32",
                "ori_shape": (1, 3, 5, 1, 16, 16),
                "ori_format": "C1HWNCoC0"
            },
            {
                "shape": (3, 5, 15, 1),
                "format": "HWCN",
                "dtype": "int32",
                "ori_shape": (3, 5, 15, 1),
                "ori_format": "HWCN"
            },
            "C1HWNCoC0",
            "HWCN",
        ],
    "case_name": 'test_depthwise6d24d_small_shape_int32',
    "expect": "success"
}

case_small_shape_uint16 = {
    "params":
        [
            {
                "shape": (1, 3, 5, 1, 16, 16),
                "format": "C1HWNCoC0",
                "dtype": "uint16",
                "ori_shape": (1, 3, 5, 1, 16, 16),
                "ori_format": "C1HWNCoC0"
            },
            {
                "shape": (3, 5, 15, 1),
                "format": "HWCN",
                "dtype": "uint16",
                "ori_shape": (3, 5, 15, 1),
                "ori_format": "HWCN"
            },
            "C1HWNCoC0",
            "HWCN"
        ],
    "case_name": 'test_depthwise6d24d_small_shape_uint16',
    "expect": "success"
}

case_c0_16_aligned_fp32 = {
    "params":
        [
            {
                "shape": (6, 3, 5, 1, 16, 16),
                "format": "C1HWNCoC0",
                "dtype": "float32",
                "ori_shape": (6, 3, 5, 1, 16, 16),
                "ori_format": "C1HWNCoC0"
            },
            {
                "shape": (3, 5, 96, 1),
                "format": "HWCN",
                "dtype": "float32",
                "ori_shape": (3, 5, 96, 1),
                "ori_format": "HWCN"
            },
            "C1HWNCoC0",
            "HWCN"
        ],
    "case_name": 'test_depthwise6d24d_c0_16_aligned_fp32',
    "expect": "success"
}

case_big_shape_fp32 = {
    "params":
        [
            {
                "shape": (16, 16, 32, 1, 16, 16),
                "format": "C1HWNCoC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 32, 1, 16, 16),
                "ori_format": "C1HWNCoC0"
            },
            {
                "shape": (16, 32, 256, 1),
                "format": "HWCN",
                "dtype": "float32",
                "ori_shape": (16, 32, 256, 1),
                "ori_format": "HWCN"
            },
            "C1HWNCoC0",
            "HWCN"
        ],
    "case_name": 'test_depthwise6d24d_big_shape_fp32',
    "expect": "success"
}

case_big_prime_fp16 = {
    "params":
        [
            {
                "shape": (1, 1731, 3, 1, 16, 16),
                "format": "C1HWNCoC0",
                "dtype": "float16",
                "ori_shape": (1, 1731, 3, 1, 16, 16),
                "ori_format": "C1HWNCoC0"
            },
            {
                "shape": (1731, 3, 15, 1),
                "format": "HWCN",
                "dtype": "float16",
                "ori_shape": (1731, 3, 15, 1),
                "ori_format": "HWCN"
            },
            "C1HWNCoC0",
            "HWCN"
        ],
    "case_name": 'test_depthwise6d24d_big_prime_fp16',
    "expect": "success"
}

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_int32)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_uint16)
ut_case.add_case(["Ascend910", "Ascend310"], case_c0_16_aligned_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_big_shape_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_big_prime_fp16)
# ut_case.add_case(["Ascend310"], case1)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)


