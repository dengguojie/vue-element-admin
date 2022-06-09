# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
test_sparse_segment_sum_grad.py
"""
import copy as cp
import numpy as np


def calc_expect_func(grad, indices, segment_ids, output_dim0, output):
    grad_shape = grad["shape"]
    grad_dtype = grad["dtype"]
    grad_value = grad["value"]
    indices_shape = indices["shape"]
    indices_value = indices["value"]
    segment_ids_shape = segment_ids["shape"]
    segment_ids_value = segment_ids["value"]
    output_dim0_value = output_dim0["value"]

    if len(indices_shape) != 1 or len(segment_ids_shape) != 1 or indices_shape[0] != segment_ids_shape[0]:
        return np.NAN

    output_shape = cp.deepcopy(grad_shape)
    output_shape[0] = output_dim0_value[0]
    y = np.zeros(output_shape).astype(grad_dtype)
    if output_dim0_value[0] == 0 or segment_ids_shape[0] == 0:
        return [y, ]

    for i, segment_idx in enumerate(segment_ids_value):
        output_idx = indices_value[i]
        if segment_idx >= grad_shape[0]:
            return np.NAN
        if output_idx >= output_dim0_value[0]:
            return np.NAN
        y[output_idx] += grad_value[segment_idx]

    return [y, ]
