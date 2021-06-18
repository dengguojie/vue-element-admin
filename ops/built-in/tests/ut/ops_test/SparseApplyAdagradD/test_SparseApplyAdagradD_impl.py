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
ut_case = OpUT("SparseApplyAdagradD", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    1.0],
         "case_name": "SparseApplyAdagradD_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("Ascend910A", case1)

def calc_expect_func(var, accum, grad, indices, var_out, accum_out,
                     lr, use_locking, update_slots):
    inputArr1_var = tf.Variable(var["value"],dtype = var["dtype"])
    inputArr2_accum = tf.Variable(accum["value"],dtype = accum["dtype"])
    inputArr3_grad = grad["value"]
    inputArr4_indices = indices["value"]
    outputArr = gen_training_ops.sparse_apply_adagrad(inputArr1_var, inputArr2_accum, lr, inputArr3_grad, inputArr4_indices, use_locking=use_locking,update_slots=update_slots)
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
               0.01, False, True],
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
               0.01, False, True],
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
               0.01, False, True],
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
               0.01, False, True],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.05, 0.05)
})
if __name__ == '__main__':
    ut_case.run("Ascend910A")

