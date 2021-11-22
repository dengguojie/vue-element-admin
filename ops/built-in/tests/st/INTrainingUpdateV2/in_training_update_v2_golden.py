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
in_training_update_v2_golden.py
"""
import os
import numpy as np


def calc_expect_func(x,
                     sum,
                     square_sum,
                     gamma=None,
                     beta=None,
                     mean=None,
                     variance=None,
                     y=None,
                     batch_mean=None,
                     batch_variance=None,
                     momentum=0.1,
                     epsilon=0.00001):
    format_x = x["format"]
    shape = x["shape"]
    dtype_x = x["dtype"]
    input_x = x["value"]

    if format_x == "NC1HWC0":
        axis = [2, 3]
    else:
        axis = [1, 3, 4]
    axis = tuple(axis)
    print("331111111111, format_x, axis, shape", format_x, axis, shape, flush=True)

    if dtype_x == "float16":
        input_x = input_x.astype(np.float32)

    num = 1
    for i in axis:
        num *= shape[i]
    current_mean = sum["value"] / num
    current_var = square_sum["value"] / num - current_mean * current_mean

    result = ((input_x - current_mean) / (np.sqrt(current_var + epsilon)))
    if gamma is not None and beta is not None:
        result = result * gamma["value"] + beta["value"]
    result_mean = current_mean

    if num == 1:
        batch_var_scalar = 0.0
    else:
        batch_var_scalar = float(num) / (num - 1)
    result_var = current_var * batch_var_scalar
    if mean is not None and variance is not None:
        factor_reverse = 1.0 - momentum
        mean_mul = result_mean * momentum
        mean_mul_rev = mean["value"] * factor_reverse
        result_mean = mean_mul + mean_mul_rev

        var_mul = result_var * momentum
        mean_var_rev = variance["value"] * factor_reverse
        result_var = var_mul + mean_var_rev

    if dtype_x == "float16":
        result = result.astype(np.float16)

    return result, result_mean, result_var
