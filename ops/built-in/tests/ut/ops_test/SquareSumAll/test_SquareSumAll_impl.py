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

SquareSumAll ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("SquareSumAll")


case_small_shape_scalar_fp32 = {
    "params":
        [
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_square_sum_all_small_shape_scalar_fp32',
    "expect": "success"
}

case_scalar_not_aligned_gt_block_size_fp32 = {
    "params":
        [
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "ND"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "ND"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "ND"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_square_sum_all_scalar_not_aligned_gt_block_size_fp32',
    "expect": "success"
}

case_aligned_multi_dim_fp32 = {
    "params":
        [
            {
                "shape": (16, 16, 64, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "ND"
            },
            {
                "shape": (16, 16, 64, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "ND"
            },
            {
                "shape": (16, 16, 64, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "ND"
            },
            {
                "shape": (16, 16, 64, 32),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_square_sum_all_aligned_multi_dim_fp32',
    "expect": "success"
}

case_scalar_big_prime_fp32 = {
    "params":
        [
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, ),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_square_sum_all_scalar_big_prime_fp32',
    "expect": "success"
}

case_big_shape_not_aligned_fp32 = {
    "params":
        [
            {
                "shape": (9973, 13, 829),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 13, 829),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 13, 829),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 13, 829),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 13, 829),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 13, 829),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 13, 829),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (9973, 13, 829),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_square_sum_all_big_shape_not_aligned_fp32',
    "expect": "success"
}

case_cut_high_dim_single_not_aligned_fp32 = {
    "params":
        [
            {
                "shape": (32, 1913),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "ND"
            },
            {
                "shape": (32, 1913),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "ND"
            },
            {
                "shape": (32, 1913),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "ND"
            },
            {
                "shape": (32, 1913),
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "ND"
            }
        ],
    "case_name": 'test_square_sum_all_cut_high_dim_single_not_aligned_fp32',
    "expect": "success"
}

def test_get_op_support_info(test_arg):
    from impl.square_sum_all import get_op_support_info
    get_op_support_info(
        {"shape": (32, 1913), "format": "ND", "dtype": "float32", "ori_shape": (32, 1913), "ori_format": "ND"},
        {"shape": (32, 1913), "format": "ND", "dtype": "float32", "ori_shape": (32, 1913), "ori_format": "ND"},
        {"shape": (32, 1913), "format": "ND", "dtype": "float32", "ori_shape": (32, 1913), "ori_format": "ND"},
        {"shape": (32, 1913), "format": "ND", "dtype": "float32", "ori_shape": (32, 1913), "ori_format": "ND"})

ut_case.add_case(["Ascend910"], case_small_shape_scalar_fp32)
ut_case.add_case(["Ascend910"], case_scalar_not_aligned_gt_block_size_fp32)
ut_case.add_case(["Ascend910"], case_aligned_multi_dim_fp32)
ut_case.add_case(["Ascend910"], case_scalar_big_prime_fp32)
ut_case.add_case(["Ascend910"], case_big_shape_not_aligned_fp32)
ut_case.add_case(["Ascend910"], case_cut_high_dim_single_not_aligned_fp32)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)


# ut_case.add_case(["Ascend310"], case1)

def calc_expect_func(input_x, input_y, output_x, output_y):
    dtype = input_x["dtype"]
    if dtype == "fp16" or dtype == "float16":
        sdtype = np.float16
    elif dtype == "fp32" or dtype == "float32":
        sdtype = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    scalar_shape = (1,)
    input_x_data = input_x["value"]
    input_y_data = input_y["value"]
    if sdtype == np.float16:
        input_x_data = input_x_data.astype(np.float32)
        input_y_data = input_y_data.astype(np.float32)

    sum_square_x = np.square(input_x_data).sum()
    sum_square_y = np.square(input_y_data).sum()

    if sdtype == np.float16:
        sum_square_x = sum_square_x.astype(sdtype)
        sum_square_y = sum_square_y.astype(sdtype)

    out_x = np.ones(scalar_shape, dtype=sdtype) * sum_square_x
    out_y = np.ones(scalar_shape, dtype=sdtype) * sum_square_y
    return [out_x, out_y]

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7,), "shape": (7,), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7,), "shape": (7,), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (279,499), "shape": (279,499), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (279,499), "shape": (279,499), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})