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

StridedSliceAssign ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("StridedSliceAssign",
               "impl.strided_slice_assign_d",
               "strided_slice_assign_d")


case_small_shape_fp16 = {
    "params":
        [
            {
                "shape": (29, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (29, ),
                "ori_format": "ND"
            },
            {
                "shape": (6, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (6, ),
                "ori_format": "ND"
            },
            {
                "shape": (29, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (29, ),
                "ori_format": "ND"
            },
            (0, ),
            (11, ),
            (1, ),
            0, 0, 0, 0, 0
        ],
    "case_name": 'test_strided_slice_assign_d_small_shape_fp16',
    "expect": "success"
}

case_small_shape_fp32 = {
    "params":
        [
            {
                "shape": (17, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (17, ),
                "ori_format": "ND"
            },
            {
                "shape": (0, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (0, ),
                "ori_format": "ND"
            },
            {
                "shape": (17, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (17, ),
                "ori_format": "ND"
            },
            (0, ),
            (11, ),
            (1, ),
            0, 0, 0, 0, 0
        ],
    "case_name": 'test_strided_slice_assign_d_small_shape_fp32',
    "expect": "success"
}

case_small_shape_int32 = {
    "params":
        [
            {
                "shape": (17, ),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (17, ),
                "ori_format": "ND"
            },
            {
                "shape": (0, ),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (0, ),
                "ori_format": "ND"
            },
            {
                "shape": (17, ),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (17, ),
                "ori_format": "ND"
            },
            (0, ),
            (11, ),
            (1, ),
            0, 0, 0, 0, 0
        ],
    "case_name": 'test_strided_slice_assign_d_small_shape_int32',
    "expect": "success"
}

case_multi_dim_not_aligned_fp32 = {
    "params":
        [
            {
                "shape": (3, 2, 4, 5, 6, 9),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (3, 2, 4, 5, 6, 9),
                "ori_format": "ND"
            },
            {
                "shape": (0, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (0, ),
                "ori_format": "ND"
            },
            {
                "shape": (3, 2, 4, 5, 6, 9),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (3, 2, 4, 5, 6, 9),
                "ori_format": "ND"
            },
            (0, 0, 0, 0, 0, 0),
            (2, 2, 2, 5, 2, 9),
            (1, 1, 1, 1, 1, 1),
            0, 0, 0, 0, 0
        ],
    "case_name": 'test_strided_slice_assign_d_multi_dim_not_aligned_fp32',
    "expect": "success"
}

case_multi_mask_fp32 = {
    "params":
        [
            {
                "shape": (32, 128),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 128),
                "ori_format": "ND"
            },
            {
                "shape": (7, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (7, ),
                "ori_format": "ND"
            },
            {
                "shape": (32, 128),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 128),
                "ori_format": "ND"
            },
            (5, 0),
            (8, 32),
            (1, 1),
            1, 1, 0, 2, 1
        ],
    "case_name": 'test_strided_slice_assign_d_multi_mask_fp32',
    "expect": "success"
}

case_strides_gt_one_fp32 = {
    "params":
        [
            {
                "shape": (32, 128, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 128, 32),
                "ori_format": "ND"
            },
            {
                "shape": (3, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (3, ),
                "ori_format": "ND"
            },
            {
                "shape": (32, 128, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 128, 32),
                "ori_format": "ND"
            },
            (0, 0, 0),
            (25, 32, 26),
            (2, 2, 1),
            0, 0, 0, 0, 0
        ],
    "case_name": 'test_strided_slice_assign_d_strides_gt_one_fp32',
    "expect": "success"
}

case_special_ellipsis_mask_fp32 = {
    "params":
        [
            {
                "shape": (32, 128, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 128, 32),
                "ori_format": "ND"
            },
            {
                "shape": (3, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (3, ),
                "ori_format": "ND"
            },
            {
                "shape": (32, 128, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 128, 32),
                "ori_format": "ND"
            },
            (0, 0, 0),
            (25, 32, 26),
            (2, 2, 1),
            0, 0, 1, 0, 0
        ],
    "case_name": 'test_strided_slice_assign_d_special_ellipsis_mask_fp32',
    "expect": "success"
}

case_begin_end_shape_diff_ref_fp32 = {
    "params":
        [
            {
                "shape": (32, 128, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 128, 32),
                "ori_format": "ND"
            },
            {
                "shape": (3, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (3, ),
                "ori_format": "ND"
            },
            {
                "shape": (32, 128, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 128, 32),
                "ori_format": "ND"
            },
            (0, 0),
            (25, 32),
            (1, 1),
            0, 0, 0, 0, 0
        ],
    "case_name": 'test_strided_slice_assign_d_begin_end_shape_diff_ref_fp32',
    "expect": "success"
}

case_big_prime_fp32 = {
    "params":
        [
            {
                "shape": (1321, 73),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1321, 73),
                "ori_format": "ND"
            },
            {
                "shape": (3, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (3, ),
                "ori_format": "ND"
            },
            {
                "shape": (1321, 73),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1321, 73),
                "ori_format": "ND"
            },
            (55, 0),
            (386, 73),
            (1, 1),
            0, 0, 0, 0, 0
        ],
    "case_name": 'test_strided_slice_assign_d_big_prime_fp32',
    "expect": "success"
}

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_int32)
ut_case.add_case(["Ascend910", "Ascend310"], case_multi_dim_not_aligned_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_multi_mask_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_strides_gt_one_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_special_ellipsis_mask_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_begin_end_shape_diff_ref_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_big_prime_fp32)


# ut_case.add_case(["Ascend310"], case1)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
