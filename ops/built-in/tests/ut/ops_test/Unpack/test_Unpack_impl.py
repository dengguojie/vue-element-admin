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


def test_import_lib(test_arg):
    import sys
    import importlib
    modulename = sys.modules.get("impl.unpack")
    importlib.reload(modulename)


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

case_shape_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (128, 28, 28, 2, 24),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (128, 28, 28, 2, 24),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (128, 28, 28, 24),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (128, 28, 28, 24),
                    "ori_format": "ND"
                },
                {
                    "shape": (128, 28, 28, 24),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (128, 28, 28, 24),
                    "ori_format": "ND"
                }
            ],
            2, 3
        ],
    "case_name": 'test_unpack_case_shape_not_aligned_fp16',
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

case_num_is_none = {
    "params":
        [
            {
                "shape": (512, 2, 3),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (512, 2, 3),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (512, 3),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (512, 3),
                    "ori_format": "ND"
                }
            ],
            None, 1
        ],
    "case_name": 'test_unpack_num_is_none_case',
    "expect": "success"
}

case_axis1_int32 = {
    "params":
        [
            {
                "shape": (42, 2, 32775),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (42, 2, 32775),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (42, 32775),
                    "format": "ND",
                    "dtype": "int32",
                    "ori_shape": (42, 32775),
                    "ori_format": "ND"
                },
                {
                    "shape": (42, 32775),
                    "format": "ND",
                    "dtype": "int32",
                    "ori_shape": (42, 32775),
                    "ori_format": "ND"
                }
            ],
            2, 1
        ],
    "case_name": 'test_unpack_axis1_int32',
    "expect": "success"
}

case_5hd_supoort = {
    "params":
        [
            {
                "shape": (1, 1, 2, 2, 16),
                "format": "NC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 2, 2, 16),
                "ori_format": "NHWC"
            },
            [
                {
                    "shape": (1, 1, 2, 2, 16),
                    "format": "NC1HWC0",
                    "dtype": "float16",
                    "ori_shape": (1, 2, 2, 16),
                    "ori_format": "NHWC"
                }
            ],
            1, 0
        ],
    "case_name": 'test_unpack_case_5hd_support_case',
    "expect": "success"
}

case_input_num_invaild = {
    "params":
        [
            {
                "shape": (512, 2, 3),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (512, 2, 3),
                "ori_format": "ND"
            },
            [
                {
                    "shape": (512, 3),
                    "format": "ND",
                    "dtype": "float16",
                    "ori_shape": (512, 3),
                    "ori_format": "ND"
                }
            ],
            10, 1
        ],
    "case_name": 'test_unpack_input_num_invaild_case',
    "expect": RuntimeError
}

def test_get_op_support_info(test_arg):
    from impl.unpack import get_op_support_info
    get_op_support_info({"shape": (20, 2, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (20, 2, 16, 16),"ori_format": "NCHW"},
                        {"shape": (20, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (20, 16, 16),"ori_format": "NCHW"},
                        2, 0)

def test_check_supported(test_arg):
    from impl.unpack import check_supported
    check_supported({"shape": (1, 4000, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 4000, 1, 1), "ori_format": "NCHW"},
                    {"shape": (1, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (1, 1, 1), "ori_format": "NCHW"},
                    4000, 1)

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_scalar_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_shape_not_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_not_aligned_fp16)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case_axis1_int32)
# ut_case.add_case(["Ascend910", "Ascend310"], case_last_dim_lt_one_block_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_last_dim_lt_one_block_2_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_direct_moved_by_ub_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_multi_dim_aligned_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_big_prime_fp16)
# ut_case.add_case(["Ascend910", "Ascend310"], case_big_shape_not_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_multi_output_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_num_is_none)
ut_case.add_case(["Ascend910", "Ascend310"], case_5hd_supoort)
ut_case.add_case(["Ascend910", "Ascend310"], case_input_num_invaild)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)
ut_case.add_cust_test_func(test_func=test_check_supported)


# ut_case.add_case(["Ascend310"], case1)
def test_op_select_format(test_arg):
    from impl.unpack import op_select_format
    op_select_format({
                "shape": (1, 1, 2, 2, 16),
                "format": "NC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 2, 2, 16),
                "ori_format": "NHWC"
            },
            [
                {
                    "shape": (1, 1, 2, 2, 16),
                    "format": "NC1HWC0",
                    "dtype": "float16",
                    "ori_shape": (1, 2, 2, 16),
                    "ori_format": "NHWC"
                }
            ],
            1, 0, "test_unpack_case_5hd_support_case")

ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_import_lib)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
