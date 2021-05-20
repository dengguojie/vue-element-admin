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

TopK ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("TopK",
               "impl.top_k",
               "top_k")


case_k_lt_16_fp16 = {
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
                "shape": (10, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (10, ),
                "ori_format": "ND"
            },
            {
                "shape": (10, ),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (10, ),
                "ori_format": "ND"
            },
            10,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_k_lt_16_fp16',
    "expect": "success"
}

case_k_lt_4096_fp16 = {
    "params":
        [
            {
                "shape": (5000, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (5000, ),
                "ori_format": "ND"
            },
            {
                "shape": (5000, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (5000, ),
                "ori_format": "ND"
            },
            {
                "shape": (3000, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (3000, ),
                "ori_format": "ND"
            },
            {
                "shape": (3000, ),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (3000, ),
                "ori_format": "ND"
            },
            3000,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_k_lt_4096_fp16',
    "expect": "success"
}

case_k_between_4096_and_5120_fp16 = {
    "params":
        [
            {
                "shape": (5000, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (5000, ),
                "ori_format": "ND"
            },
            {
                "shape": (5000, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (5000, ),
                "ori_format": "ND"
            },
            {
                "shape": (4500, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (4500, ),
                "ori_format": "ND"
            },
            {
                "shape": (4500, ),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (4500, ),
                "ori_format": "ND"
            },
            4500,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_k_between_4096_and_5120_fp16',
    "expect": "success"
}

case_k_not_aligned_gt_one_block_fp16 = {
    "params":
        [
            {
                "shape": (63, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (63, ),
                "ori_format": "ND"
            },
            {
                "shape": (63, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (63, ),
                "ori_format": "ND"
            },
            {
                "shape": (10, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (10, ),
                "ori_format": "ND"
            },
            {
                "shape": (10, ),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (10, ),
                "ori_format": "ND"
            },
            10,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_not_aligned_gt_one_block_fp16',
    "expect": "success"
}

case_k_big_shape_not_aligned_fp16 = {
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
                "shape": (5000, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (5000, ),
                "ori_format": "ND"
            },
            {
                "shape": (5000, ),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (5000, ),
                "ori_format": "ND"
            },
            5000,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_big_shape_not_aligned_fp16',
    "expect": "success"
}

case_k_last_dim_lt_4096_fp16 = {
    "params":
        [
            {
                "shape": (32, 3600),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (32, 3600),
                "ori_format": "ND"
            },
            {
                "shape": (32, 3600),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (32, 3600),
                "ori_format": "ND"
            },
            {
                "shape": (32, 2000),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (32, 2000),
                "ori_format": "ND"
            },
            {
                "shape": (32, 2000),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (32, 2000),
                "ori_format": "ND"
            },
            2000,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_last_dim_lt_4096_fp16',
    "expect": "success"
}

case_k_last_dim_gt_4096_fp16 = {
    "params":
        [
            {
                "shape": (32, 5100),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (32, 5100),
                "ori_format": "ND"
            },
            {
                "shape": (32, 5100),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (32, 5100),
                "ori_format": "ND"
            },
            {
                "shape": (32, 2000),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (32, 2000),
                "ori_format": "ND"
            },
            {
                "shape": (32, 2000),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (32, 2000),
                "ori_format": "ND"
            },
            2000,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_last_dim_gt_4096_fp16',
    "expect": "success"
}

case_k_multi_dim_aligned_fp16 = {
    "params":
        [
            {
                "shape": (16, 32, 4096),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 4096),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 4096),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 4096),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 3000),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 3000),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 3000),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (16, 32, 3000),
                "ori_format": "ND"
            },
            3000,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_multi_dim_aligned_fp16',
    "expect": "success"
}

case_k_multi_core_single_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (16, 32, 4095),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 4095),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 4095),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 4095),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 3000),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 3000),
                "ori_format": "ND"
            },
            {
                "shape": (16, 32, 3000),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (16, 32, 3000),
                "ori_format": "ND"
            },
            3000,
            False,
            -1,
            True
        ],
    "case_name": 'test_top_k_multi_core_single_not_aligned_fp16',
    "expect": "success"
}

ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_lt_16_fp16)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_lt_4096_fp16)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_between_4096_and_5120_fp16)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_not_aligned_gt_one_block_fp16)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_big_shape_not_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_last_dim_lt_4096_fp16)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_last_dim_gt_4096_fp16)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_multi_dim_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend920A"], case_k_multi_core_single_not_aligned_fp16)

def test_1981(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.top_k import top_k
    te_set_version("Ascend920A", "VectorCore")
    top_k(*(case_k_lt_16_fp16["params"]))

ut_case.add_cust_test_func(test_func=test_1981)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
