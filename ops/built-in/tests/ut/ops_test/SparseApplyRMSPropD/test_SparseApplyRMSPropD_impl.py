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

SparseApplyRMSPropD ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
import random

ut_case = OpUT("SparseApplyRmsPropD",
               "impl.sparse_apply_rms_prop_d",
               "sparse_apply_rms_prop_d")

case_small_shape_scalar_fp32 = {
    "params":
        [
            {
                "shape": (1, ),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # lr
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1,),  # grad
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),   # indices
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1, ),  # var
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            0.5, 0.99,
            0.0001,
            False
        ],
    "case_name": 'test_sparse_apply_rms_prop_d_small_shape_scalar_fp32',
    "expect": "success"
}

case_not_aligned_gt_one_block_fp32 = {
    "params":
        [
            {
                "shape": (33, ),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (33, ),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (33, ),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # lr
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (10,),  # grad
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (10, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (10, ),   # indices
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (10, ),
                "ori_format": "ND"
            },
            {
                "shape": (33, ),  # var
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (33, ),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (33, ),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            0.5, 0.99,
            0.0001,
            False
        ],
    "case_name": 'test_sparse_apply_rms_prop_d_not_aligned_gt_one_block_fp32',
    "expect": "success"
}

case_multi_dim_aligned_fp32 = {
    "params":
        [
            {
                "shape": (16, 16, 64, 32),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # lr
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (10, 16, 64, 32),  # grad
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (10, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (10, ),   # indices
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (10, ),
                "ori_format": "ND"
            },
            {
                "shape": (16, 16, 64, 32),  # var
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            0.5, 0.99,
            0.0001,
            False
        ],
    "case_name": 'test_sparse_apply_rms_prop_d_multi_dim_aligned_fp32',
    "expect": "success"
}

case_big_prime_scalar_fp32 = {
    "params":
        [
            {
                "shape": (9973, ),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, ),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, ),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # lr
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (4317,),  # grad
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (4317, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (4317, ),   # indices
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (4317, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, ),  # var
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (9973, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, ),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, ),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "NC1HWC0"
            },
            0.5, 0.99,
            0.0001,
            False
        ],
    "case_name": 'test_sparse_apply_rms_prop_d_big_prime_scalar_fp32',
    "expect": "success"
}

case_big_shape_not_aligned_fp32 = {
    "params":
        [
            {
                "shape": (9973, 13, 8397),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8397),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8397),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # lr
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (1631, 13, 8397),  # grad
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1631, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1631, ),   # indices
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (1631, ),
                "ori_format": "ND"
            },
            {
                "shape": (9973, 13, 8397),  # var
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8397),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8397),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            0.5, 0.99,
            0.0001,
            False
        ],
    "case_name": 'test_sparse_apply_rms_prop_d_big_shape_not_aligned_fp32',
    "expect": "success"
}

case_multi_core_single_not_aligned_fp32 = {
    "params":
        [
            {
                "shape": (32, 1913),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (32, 1913),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (32, 1913),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # lr
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (16, 1913),  # grad
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, ),   # indices
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (16, ),
                "ori_format": "ND"
            },
            {
                "shape": (32, 1913),  # var
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (32, 1913),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (32, 1913),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            0.5, 0.99,
            0.0001,
            False
        ],
    "case_name": 'test_sparse_apply_rms_prop_d_multi_core_single_not_aligned_fp32',
    "expect": "success"
}

case_multi_dim_aligned_indices_int64 = {
    "params":
        [
            {
                "shape": (16, 16, 64, 32),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # lr
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "ND"
            },
            {
                "shape": (10, 16, 64, 32),  # grad
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (10, ),   # indices
                "format": "ND",
                "dtype": "int64",
                "ori_shape": (10, ),
                "ori_format": "ND"
            },
            {
                "shape": (16, 16, 64, 32),  # var
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # ms
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # mom
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            0.5, 0.99,
            0.0001,
            False
        ],
    "case_name": 'test_sparse_apply_rms_prop_d_multi_dim_aligned_indices_int64',
    "expect": "success"
}

ut_case.add_case(["Ascend910A"], case_small_shape_scalar_fp32)
ut_case.add_case(["Ascend910A"], case_not_aligned_gt_one_block_fp32)
ut_case.add_case(["Ascend910A"], case_multi_dim_aligned_fp32)
ut_case.add_case(["Ascend910A"], case_big_prime_scalar_fp32)
ut_case.add_case(["Ascend910A"], case_big_shape_not_aligned_fp32)
ut_case.add_case(["Ascend910A"], case_multi_core_single_not_aligned_fp32)
ut_case.add_case(["Ascend910A"], case_multi_dim_aligned_indices_int64)


def calc_expect_func(var, ms, mom, lr, grad, indices, out_var,
                     out_ms, out_mom, rho, momentum, epsilon, use_locking=False):
    var_data = var["value"]
    ms_data = ms["value"]
    mom_data = mom["value"]
    lr = lr["value"]
    grad = grad["value"]
    indices = indices["value"]

    for i, idx in enumerate(indices):
        ms_data[idx] = ms_data[idx] * rho + grad[i] * grad[i] * (1 - rho)
        mom_data[idx] = mom_data[idx] * momentum + (lr[0] * grad[i]) / np.sqrt(ms_data[idx] + epsilon)
        var_data[idx] = var_data[idx] - mom_data[idx]

    return [var_data, ms_data, mom_data]

ut_case.add_precision_case("Ascend910A", {
    "params": [{"shape": (16, 16, 32), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (16, 16, 32), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (16, 16, 32), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (16, 16, 32), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (16, 16, 32), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (16, 16, 32), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (1, ), "format": "ND", "dtype": "float32", "ori_shape": (1, ), "ori_format": "ND", "param_type": "input"},
               {"shape": (16, 16, 32), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (16, 16, 32), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (16, ), "format": "ND", "dtype": "int64", "ori_shape": (16, ), "ori_format": "ND", "param_type": "input", "value":np.array(random.sample(range(16), 16)).astype("int64")},
               {"shape": (16, 16, 32), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (16, 16, 32), "ori_format": "NC1HWC0", "param_type": "output"},
               {"shape": (16, 16, 32), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (16, 16, 32), "ori_format": "NC1HWC0", "param_type": "output"},
               {"shape": (16, 16, 32), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (16, 16, 32), "ori_format": "NC1HWC0", "param_type": "output"},
               0.5, 0.99, 0.0001, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
ut_case.add_precision_case("Ascend910A", {
    "params": [{"shape": (4, 1, 317), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (4, 1, 317), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (4, 1, 317), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (4, 1, 317), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (4, 1, 317), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (4, 1, 317), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (1, ), "format": "ND", "dtype": "float32", "ori_shape": (1, ), "ori_format": "ND", "param_type": "input"},
               {"shape": (4, 1, 317), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (4, 1, 317), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (4, ), "format": "ND", "dtype": "int64", "ori_shape": (4, ), "ori_format": "ND", "param_type": "input", "value":np.array(random.sample(range(4), 4)).astype("int64")},
               {"shape": (4, 1, 317), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (4, 1, 317), "ori_format": "NC1HWC0", "param_type": "output"},
               {"shape": (4, 1, 317), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (4, 1, 317), "ori_format": "NC1HWC0", "param_type": "output"},
               {"shape": (4, 1, 317), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (4, 1, 317), "ori_format": "NC1HWC0", "param_type": "output"},
               0.5, 0.99, 0.0001, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"shape": (128, 1, 64), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (128, 1, 64), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (128, 1, 64), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (128, 1, 64), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (128, 1, 64), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (128, 1, 64), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (1, ), "format": "ND", "dtype": "float32", "ori_shape": (1, ), "ori_format": "ND", "param_type": "input"},
               {"shape": (128, 1, 64), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (128, 1, 64), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (128, ), "format": "ND", "dtype": "int64", "ori_shape": (128, ), "ori_format": "ND", "param_type": "input", "value":np.array(random.sample(range(128), 128)).astype("int64")},
               {"shape": (128, 1, 64), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (128, 1, 64), "ori_format": "NC1HWC0", "param_type": "output"},
               {"shape": (128, 1, 64), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (128, 1, 64), "ori_format": "NC1HWC0", "param_type": "output"},
               {"shape": (128, 1, 64), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (128, 1, 64), "ori_format": "NC1HWC0", "param_type": "output"},
               0.5, 0.99, 0.0001, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"shape": (14, 1, 8, 16), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (14, 1, 8, 16), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (14, 1, 8, 16), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (14, 1, 8, 16), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (14, 1, 8, 16), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (14, 1, 8, 16), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (1, ), "format": "ND", "dtype": "float32", "ori_shape": (1, ), "ori_format": "ND", "param_type": "input"},
               {"shape": (14, 1, 8, 16), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (14, 1, 8, 16), "ori_format": "NC1HWC0", "param_type": "input"},
               {"shape": (14, ), "format": "ND", "dtype": "int64", "ori_shape": (14, ), "ori_format": "ND", "param_type": "input", "value":np.array(random.sample(range(14), 14)).astype("int64")},
               {"shape": (14, 1, 8, 16), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (14, 1, 8, 16), "ori_format": "NC1HWC0", "param_type": "output"},
               {"shape": (14, 1, 8, 16), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (14, 1, 8, 16), "ori_format": "NC1HWC0", "param_type": "output"},
               {"shape": (14, 1, 8, 16), "format": "NC1HWC0", "dtype": "float32", "ori_shape": (14, 1, 8, 16), "ori_format": "NC1HWC0", "param_type": "output"},
               0.5, 0.99, 0.0001, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
