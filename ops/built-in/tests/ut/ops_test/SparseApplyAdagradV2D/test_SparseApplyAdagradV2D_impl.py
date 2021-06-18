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

SparseApplyAdagradV2D ut case
"""
import numpy as np
from op_test_frame.common import precision_info
import tensorflow as tf
from tensorflow.python.training import gen_training_ops
from op_test_frame.ut import OpUT
import random

ut_case = OpUT("SparseApplyAdagradV2D",
               "impl.sparse_apply_adagrad_v2_d",
               "sparse_apply_adagrad_v2_d")

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
                "shape": (1, ),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1,),  # grad
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),   # indices
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # var
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1, ),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            # lr,epsilon,use_locking,update_slots
            0.01,
            0.0001,
            False,
            False
        ],
    "case_name": 'test_sparse_apply_adagrad_v2_d_small_shape_scalar_fp32',
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
                "shape": (33, ),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (10, ),  # grad
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (10, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (10, ),   # indices
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (10, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (33, ),  # var
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (33, ),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (33, ),
                "ori_format": "NC1HWC0"
            },
            # lr,epsilon,use_locking,update_slots
            0.01,
            0.0001,
            False,
            False
        ],
    "case_name": 'test_sparse_apply_adagrad_v2_d_not_aligned_gt_one_block_fp32',
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
                "shape": (16, 16, 64, 32),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (10, 16, 64, 32),  # grad
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (10, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (10, ),   # indices
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (10, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # var
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 16, 64, 32),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, 16, 64, 32),
                "ori_format": "NC1HWC0"
            },
            # lr,epsilon,use_locking,update_slots
            0.01,
            0.0001,
            False,
            False
        ],
    "case_name": 'test_sparse_apply_adagrad_v2_d_multi_dim_aligned_fp32',
    "expect": "success"
}

case_big_prime_scalar_fp32 = {
    "params":
        [
            {
                "shape": (9973, ),  # var
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, ),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (4317, ),  # grad
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (4317, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (4317, ),   # indices
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (1, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, ),  # var
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (9973, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, ),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, ),
                "ori_format": "NC1HWC0"
            },
            # lr,epsilon,use_locking,update_slots
            0.01,
            0.0001,
            False,
            False
        ],
    "case_name": 'test_sparse_apply_adagrad_v2_d_big_prime_scalar_fp32',
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
                "shape": (9973, 13, 8397),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1631, 13, 8397),  # grad
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (1631, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (1631, ),   # indices
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (1631, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8397),  # var
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (9973, 13, 8397),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (9973, 13, 8397),
                "ori_format": "NC1HWC0"
            },
            # lr,epsilon,use_locking,update_slots
            0.01,
            0.0001,
            False,
            False
        ],
    "case_name": 'test_sparse_apply_adagrad_v2_d_big_shape_not_aligned_fp32',
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
                "shape": (32, 1913),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, 1913),  # grad
                "format": "ND",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (16, ),   # indices
                "format": "NC1HWC0",
                "dtype": "int32",
                "ori_shape": (16, ),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (32, 1913),  # var
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            {
                "shape": (32, 1913),  # accum
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 1913),
                "ori_format": "NC1HWC0"
            },
            # lr,epsilon,use_locking,update_slots
            0.01,
            0.0001,
            False,
            False
        ],
    "case_name": 'test_sparse_apply_adagrad_v2_d_multi_core_single_not_aligned_fp32',
    "expect": "success"
}


ut_case.add_case(["Ascend910A"], case_small_shape_scalar_fp32)
ut_case.add_case(["Ascend910A"], case_not_aligned_gt_one_block_fp32)
ut_case.add_case(["Ascend910A"], case_multi_dim_aligned_fp32)
ut_case.add_case(["Ascend910A"], case_big_prime_scalar_fp32)
ut_case.add_case(["Ascend910A"], case_big_shape_not_aligned_fp32)
ut_case.add_case(["Ascend910A"], case_multi_core_single_not_aligned_fp32)

def calc_expect_func(var, accum, grad, indices, var_out, accum_out,
                     lr, epsilon, use_locking, update_slots):
    inputArr1_var = tf.Variable(var["value"],dtype = var["dtype"])
    inputArr2_accum = tf.Variable(accum["value"],dtype = accum["dtype"])
    inputArr3_grad = grad["value"]
    inputArr4_indices = indices["value"]
    outputArr = gen_training_ops.sparse_apply_adagrad_v2(inputArr1_var, inputArr2_accum, lr,epsilon, inputArr3_grad,
                                                         inputArr4_indices, use_locking=use_locking,update_slots=update_slots)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputArr = sess.run(outputArr)
        outputArr_accum = sess.run(inputArr2_accum)

    return [outputArr, outputArr_accum]

ut_case.add_precision_case("Ascend910A", {
    'params': [{'shape': (33, 16), 'dtype': 'float32', 'ori_shape': (33, 16), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, 16), 'dtype': 'float32', 'ori_shape': (33, 16), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, 16), 'dtype': 'float32', 'ori_shape': (33, 16), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, ), 'dtype': 'int32', 'ori_shape': (33, ), 'ori_format': 'ND', 'format': 'ND', "param_type": "input", "value":np.array(random.sample(range(33), 33)).astype("int32")},
               {'shape': (33, 16), 'dtype': 'float32', 'ori_shape': (33, 16), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"},
               {'shape': (33, 16), 'dtype': 'float32', 'ori_shape': (33, 16), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"},
               0.1, 0.01, False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.05, 0.05)
})

ut_case.add_precision_case("Ascend910A", {
    'params': [{'shape': (33, 77), 'dtype': 'float32', 'ori_shape': (33, 77), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, 77), 'dtype': 'float32', 'ori_shape': (33, 77), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, 77), 'dtype': 'float32', 'ori_shape': (33, 77), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, ), 'dtype': 'int32', 'ori_shape': (33, ), 'ori_format': 'ND', 'format': 'ND', "param_type": "input", "value":np.array(random.sample(range(33), 33)).astype("int32")},
               {'shape': (33, 77), 'dtype': 'float32', 'ori_shape': (33, 77), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"},
               {'shape': (33, 77), 'dtype': 'float32', 'ori_shape': (33, 77), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"},
               0.1, 0.01, False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.05, 0.05)
})

ut_case.add_precision_case("Ascend910A", {
    'params': [{'shape': (77, 3, 6, 3), 'dtype': 'float32', 'ori_shape': (77, 3, 6, 3), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (77, 3, 6, 3), 'dtype': 'float32', 'ori_shape': (77, 3, 6, 3), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (77, 3, 6, 3), 'dtype': 'float32', 'ori_shape': (77, 3, 6, 3), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (77, ), 'dtype': 'int32', 'ori_shape': (77, ), 'ori_format': 'ND', 'format': 'ND', "param_type": "input", "value":np.array(random.sample(range(77), 77)).astype("int32")},
               {'shape': (77, 3, 6, 3), 'dtype': 'float32', 'ori_shape': (77, 3, 6, 3), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"},
               {'shape': (77, 3, 6, 3), 'dtype': 'float32', 'ori_shape': (77, 3, 6, 3), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"},
               0.1, 0.01, False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.05, 0.05)
})

ut_case.add_precision_case("Ascend910A", {
    'params': [{'shape': (32, 3, 6, 32), 'dtype': 'float32', 'ori_shape': (32, 3, 6, 32), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (32, 3, 6, 32), 'dtype': 'float32', 'ori_shape': (32, 3, 6, 32), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (32, 3, 6, 32), 'dtype': 'float32', 'ori_shape': (32, 3, 6, 32), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (32, ), 'dtype': 'int32', 'ori_shape': (32, ), 'ori_format': 'ND', 'format': 'ND', "param_type": "input", "value":np.array(random.sample(range(32), 32)).astype("int32")},
               {'shape': (32, 3, 6, 32), 'dtype': 'float32', 'ori_shape': (32, 3, 6, 32), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"},
               {'shape': (32, 3, 6, 32), 'dtype': 'float32', 'ori_shape': (32, 3, 6, 32), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"},
               0.1, 0.01, False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.05, 0.05)
})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
