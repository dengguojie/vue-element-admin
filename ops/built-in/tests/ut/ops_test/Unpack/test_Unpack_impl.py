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

Unpack ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("Unpack")


case_small_shape_scalar_fp16 = {
    "params":
        [
            {
                "shape": (3, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (3, ),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                },
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            3, 0
        ],
    "case_name": 'test_unpack_small_shape_scalar_fp16',
    "expect": "success"
}

case_small_shape_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (17, 33),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (17, 33),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                },
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            17, 0
        ],
    "case_name": 'test_unpack_small_shape_not_aligned_fp16',
    "expect": "success"
}

case_last_dim_lt_one_block_fp16 = {
    "params":
        [
            {
                "shape": (69, 3),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (69, 3),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                },
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            69, 0
        ],
    "case_name": 'test_unpack_last_dim_lt_one_block_fp16',
    "expect": "success"
}

case_last_dim_lt_one_block_2_fp16 = {
    "params":
        [
            {
                "shape": (810, 4),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (810, 4),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                },
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            4, 1
        ],
    "case_name": 'test_unpack_last_dim_lt_one_block_2_fp16',
    "expect": "success"
}

case_direct_moved_by_ub_fp16 = {
    "params":
        [
            {
                "shape": (1, 123, 39),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, 123, 39),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            1, 0
        ],
    "case_name": 'test_unpack_direct_moved_by_ub_fp16',
    "expect": "success"
}

case_multi_dim_aligned_fp16 = {
    "params":
        [
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            16, 0
        ],
    "case_name": 'test_unpack_multi_dim_aligned_fp16',
    "expect": "success"
}

case_big_prime_fp16 = {
    "params":
        [
            {
                "shape": (17, 9973),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (17, 9973),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                },
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            17, 0
        ],
    "case_name": 'test_unpack_big_prime_fp16',
    "expect": "success"
}

case_big_shape_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (9973, 13, 8297),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, 13, 8297),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            13, 1
        ],
    "case_name": 'test_unpack_big_shape_not_aligned_fp16',
    "expect": "success"
}

case_multi_output_fp16 = {
    "params":
        [
            {
                "shape": (512, 3174),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (512, 3174),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (3, ),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (3, ),
                    "ori_format": "ND"
                }
            ],
            512, 0
        ],
    "case_name": 'test_unpack_multi_output_fp16',
    "expect": RuntimeError
}

def test_get_op_support_info(test_arg):
    from impl.unpack import get_op_support_info
    get_op_support_info({"shape": (20, 2, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (20, 2, 16, 16),"ori_format": "NCHW"},
                        {"shape": (20, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (20, 16, 16),"ori_format": "NCHW"},
                        2, 0)

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_scalar_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_not_aligned_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_last_dim_lt_one_block_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_last_dim_lt_one_block_2_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_direct_moved_by_ub_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_multi_dim_aligned_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_big_prime_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_big_shape_not_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_multi_output_fp16)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)


# ut_case.add_case(["Ascend310"], case1)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
