# Copyright 2021 Huawei Technologies Co., Ltd
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
test_l2_normalize_grad_golden.py
"""
import numpy as np


def calc_expect_func(x, y, dy, dx, dim, eps=1e-12):
    """
    calc_expect_func
    """
    x1_value = x.get("value")
    x2_value = y.get("value")
    x3_value = dy.get("value")
    dim = tuple(dim)

    x_square = x1_value * x1_value
    x_l2norm = np.sum(x_square, dim)
    x_l2norm = x_l2norm ** 0.5
    x_l2norm = np.maximum(x_l2norm, eps)

    y_mul_dy = x2_value * x3_value
    sum_y_mul_dy = np.sum(y_mul_dy, dim)

    numerator = x3_value - x2_value * sum_y_mul_dy
    result = numerator / x_l2norm
    return [result]
