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
gen LNDropoutGrad golden
"""
import numpy as np


def _reshape_for_nz(variance, mean, gamma, mean_shape, gamma_shape):
    if len(mean_shape) > 2:
        mean_shape_new = (mean_shape[0], 1, mean_shape[1] // 16, 16, 1)
        gamma_shape_new = (1, gamma_shape[0] // 16, 1, 1, 16)
    else:
        mean_shape_new = (1, mean_shape[0] // 16, 16, 1)
        gamma_shape_new = (gamma_shape[0] // 16, 1, 1, 16)
    variance = variance.reshape(mean_shape_new)
    mean = mean.reshape(mean_shape_new)
    gamma = gamma.reshape(gamma_shape_new)

    return variance, mean, gamma


def _ln_dropout_grad_by_numpy(dy, x, variance, mean, gamma, mask, keep_prob,
                              act_format, mean_shape, gamma_shape, is_cast):
    EPSLON = 1e-12
    if len(mean_shape) > 2:
        param_axis = (0, 1)
        reduce_axis = (2)
    else:
        param_axis = (0)
        reduce_axis = (1)
    m = gamma_shape[0]
    variance = np.array(variance)
    mean = np.array(mean)
    gamma = np.array(gamma)
    if act_format == "FRACTAL_NZ":
        if len(mean_shape) > 2:
            param_axis = (0, 2, 3)
            reduce_axis = (1, 4)
        else:
            param_axis = (1, 2)
            reduce_axis = (0, 3)
        variance, mean, gamma = _reshape_for_nz(variance, mean, gamma, mean_shape, gamma_shape)

    dy = dy.astype(np.float32)
    x = x.astype(np.float32)
    variance = variance.astype(np.float32)
    mean = mean.astype(np.float32)
    gamma = gamma.astype(np.float32)
    mask = mask.astype(np.float16)
    pd_xl = dy * gamma
    sub_x_mean = x - mean
    var_elta_2 = np.power((variance + EPSLON), (-0.5))

    pd_var = np.sum(pd_xl * sub_x_mean, reduce_axis, keepdims=True) * var_elta_2 * var_elta_2 * var_elta_2 * (-0.5)
    pd_mean = np.sum(pd_xl, reduce_axis, keepdims=True) * var_elta_2 * (-1.0)
    pd_x = pd_xl * var_elta_2 + pd_var * (2.0 / m) * sub_x_mean + pd_mean * (1.0 / m)

    pd_x_dropout = pd_x * mask * (1 / keep_prob)
    pd_gamma = np.sum(dy * sub_x_mean * var_elta_2, param_axis, keepdims=True)
    pd_beta = np.sum(dy, param_axis, keepdims=True)

    pd_gamma = pd_gamma.astype(np.float32)
    pd_beta = pd_beta.astype(np.float32)
    if is_cast:
        pd_x_dropout = pd_x_dropout.astype(np.float16)
        pd_x = pd_x.astype(np.float16)

    return pd_x, pd_x_dropout, pd_gamma, pd_beta


def calc_expect_func(dy, x, variance, mean, gamma, mask, pd_x, pd_x_after_dropout, pg_gamma, pg_beta, keep_prob):
    is_cast = False
    if dy["dtype"] == "float16":
        is_cast = True
    pd_x, pd_x_dropout, pd_gamma, pd_beta = _ln_dropout_grad_by_numpy(dy["value"], x["value"], variance["value"],
                                                                      mean["value"], gamma["value"], mask["value"],
                                                                      keep_prob,
                                                                      dy["format"], mean["shape"], gamma["shape"],
                                                                      is_cast)
    return [pd_x, pd_x_dropout, pd_gamma, pd_beta]

