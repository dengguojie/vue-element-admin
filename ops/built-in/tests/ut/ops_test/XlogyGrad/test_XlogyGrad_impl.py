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

XlogyGrad ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("XlogyGrad")

def produce_shapes(shape1, shape2):
    """
    two input shapes produce three output shape
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    flag = 0
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        flag = 1

    output_shape_len = len(shape1)
    dec = output_shape_len - len(shape2)
    for i in range(dec):
        shape2 = [1] + shape2

    if flag == 1:
        shape1, shape2 = shape2, shape1

    return shape1, shape2

def broadcast_gradient_args(x,y):
    rx = []
    ry = []
    for i in range(len(x)):
        if x[i] < y[i]:
            rx.append(i)
        elif x[i] > y[i]:
            ry.append(i)
    return rx, ry

def calc_expect_func(x1, x2, grad, y1, y2):
    shape1, shape2 = produce_shapes(x1['shape'], x2['shape'])
    rx, ry = broadcast_gradient_args(shape1, shape2)
    m1 = np.array(x1['value'])
    m2 = np.nonzero(m1)
    m1[m2] = 1
    log_Arr2 = np.log(x2['value'])
    mul_Arr1 = np.multiply(m1, log_Arr2)
    grad_Arr1 = np.multiply(mul_Arr1, grad['value'])
    outputArr1 = np.sum(grad_Arr1, tuple(rx)).astype(y1['dtype'])
    div_Arr2 = np.divide(x1['value'], x2['value'])
    grad_Arr2 = np.multiply(div_Arr2, grad['value'])
    outputArr2 = np.sum(grad_Arr2, tuple(ry)).astype(y2['dtype'])
    return outputArr1, outputArr2

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
    "case_name": 'test_xlogy_grad_small_shape_scalar_fp16',
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
    "case_name": 'test_xlogy_grad_not_aligned_gt_one_block_scalar_fp16',
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
    "case_name": 'test_xlogy_grad_multi_dim_aligned_fp16',
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
    "case_name": 'test_xlogy_grad_big_scalar_scalar_fp16',
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
    "case_name": 'test_xlogy_grad_big_shape_not_aligned_fp16',
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
    "case_name": 'test_xlogy_grad_multi_core_single_not_aligned_fp16',
    "expect": "success"
}


ut_case.add_case(["Ascend910A"], case_small_shape_scalar_fp16)
ut_case.add_case(["Ascend910A"], case_not_aligned_gt_one_block_scalar_fp16)
ut_case.add_case(["Ascend910A"], case_multi_dim_aligned_fp16)
ut_case.add_case(["Ascend910A"], case_big_prime_scalar_scalar_fp16)
ut_case.add_case(["Ascend910A"], case_big_shape_not_aligned_fp16)
ut_case.add_case(["Ascend910A"], case_multi_core_single_not_aligned_fp16)


precision_case1 = {
    "params":
        [
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND",
                "param_type": "output"
            },
            {
                "shape": (33, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (33, ),
                "ori_format": "ND",
                "param_type": "output"
            }
        ],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}

precision_case2 = {
    "params":
        [
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND",
                "param_type": "output"
            },
            {
                "shape": (1, ),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (1, ),
                "ori_format": "ND",
                "param_type": "output"
            }
        ],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}
precision_case3 = {
    "params":
        [
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND",
                "param_type": "output"
            },
            {
                "shape": (16, 32, 16),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 32, 16),
                "ori_format": "ND",
                "param_type": "output"
            }
        ],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}
ut_case.add_precision_case("Ascend910A", precision_case1)
ut_case.add_precision_case("Ascend910A", precision_case2)
ut_case.add_precision_case("Ascend910A", precision_case3)

