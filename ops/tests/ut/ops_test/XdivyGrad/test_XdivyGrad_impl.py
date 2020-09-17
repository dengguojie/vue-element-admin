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

XdivyGrad ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("XdivyGrad")


case_small_shape_scalar_fp16 = {
    "params":
        [
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_xdivy_grad_small_shape_scalar_fp16',
    "expect": "success"
}

case_not_aligned_gt_one_block_scalar_fp16 = {
    "params":
        [
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_xdivy_grad_not_aligned_gt_one_block_scalar_fp16',
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
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_xdivy_grad_multi_dim_aligned_fp16',
    "expect": "success"
}

case_big_prime_scalar_scalar_fp16 = {
    "params":
        [
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_xdivy_grad_big_scalar_scalar_fp16',
    "expect": "success"
}

case_big_shape_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 8297),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (9973, 8297),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_xdivy_grad_big_shape_not_aligned_fp16',
    "expect": "success"
}

case_multi_core_single_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (1024, 31744),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1024, 31744),
                "ori_format": "ND"
            },
            {
                "shape": (1024, 31744),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1024, 31744),
                "ori_format": "ND"
            },
            {
                "shape": (1024, 31744),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1024, 31744),
                "ori_format": "ND"
            },
            {
                "shape": (1024, 31744),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1024, 31744),
                "ori_format": "ND"
            },
            {
                "shape": (1024, 31744),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1024, 31744),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_xdivy_grad_multi_core_single_not_aligned_fp16',
    "expect": "success"
}


ut_case.add_case(["Ascend910"], case_small_shape_scalar_fp16)
ut_case.add_case(["Ascend910"], case_not_aligned_gt_one_block_scalar_fp16)
ut_case.add_case(["Ascend910"], case_multi_dim_aligned_fp16)
ut_case.add_case(["Ascend910"], case_big_prime_scalar_scalar_fp16)
ut_case.add_case(["Ascend910"], case_big_shape_not_aligned_fp16)
ut_case.add_case(["Ascend910"], case_multi_core_single_not_aligned_fp16)


# ut_case.add_case(["Ascend310"], case1)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
