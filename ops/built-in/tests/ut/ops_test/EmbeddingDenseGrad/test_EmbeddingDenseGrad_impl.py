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

EmbeddingDenseGrad ut case
"""

# -*- coding:utf-8 -*-
import numpy as np
from op_test_frame.ut import ElementwiseOpUT
import os

ut_case = ElementwiseOpUT("embedding_dense_grad")

def calc_expect_func(grad, indices, y, num_weights, padding_idx, scale_by_freq):
    grad_value = grad.get('value')
    indices_value = indices.get('value')
    grad_shape = grad.get('shape')
    embedding_dim = grad_shape[-1]
    grad_value = grad_value.reshape(-1, embedding_dim)
    counts = [0] * num_weights
    indices_value = indices_value.flatten()
    for n in indices_value:
        counts[n] += 1
    res = [[0 for j in range(grad_shape[-1])] for i in range(num_weights)]
    for i in range(len(grad_value)):
        if counts[indices_value[i]] != padding_idx:
            for j in range(len(grad_value[i])):
                if scale_by_freq:
                    res[indices_value[i]][j] += grad_value[i][j] / counts[indices_value[i]]
                else:
                    res[indices_value[i]][j] += grad_value[i][j]
    res = np.array(res)
    return res

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 32), "shape": (3, 3, 32),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (23, 32), "shape": (23, 32),
                "param_type": "output"},
               23, -1, False],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 1), "shape": (3, 3, 1),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (23, 1), "shape": (23, 1),
                "param_type": "output"},
               23, -1, False],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7000, 1), "shape": (7000, 1),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (7000,), "shape": (7000,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7000, 1), "shape": (7000, 1),
                "param_type": "output"},
               7000, -1, False],
    "calc_expect_func": calc_expect_func
})

if __name__ == '__main__':
    user_home_path = os.path.expanduser('~')
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B15/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode='pv', simulator_lib_path=simulator_lib_path)

