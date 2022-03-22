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
test_spence.py
"""
import numpy as np


class Constant:
    """
    the class for constant.
    """
    SCALAR_PI = 3.1415926535897932384626433832795
    SCALAR_PI206 = SCALAR_PI * SCALAR_PI / 6.0

    SCALAR_A = (4.65128586073990045278E-5, 7.31589045238094711071E-3,
                1.33847639578309018650E-1, 8.79691311754530315341E-1,
                2.71149851196553469920E0, 4.25697156008121755724E0,
                3.29771340985225106936E0, 1.00000000000000000126E0)
    SCALAR_B = (6.90990488912553276999E-4, 2.54043763932544379113E-2,
                2.82974860602568089943E-1, 1.41172597751831069617E0,
                3.63800533345137075418E0, 5.03278880143316990390E0,
                3.54771340985225096217E0, 9.99999999999999998740E-1)

    A_COUNT = 7
    B_COUNT = 7


def polevl(data_x, coef, num):
    """
    y = polevl( x, coef, num );
    DESCRIPTION:
    Evaluates polynomial of degree N:
                        2          N
    y  =  C  + C x + C x  +...+ C x
             0    1     2          N
    Coefficients are stored in reverse order:
    coef[0] = C  , ..., coef[N] = C  .
                 N                   0
    Parameters:
    ----------
    data_x : the placeholder of data input
    coef : coef of the data
    iter_n : number of the coef
     Returns : A Tensor. Has the same type as data.
    -------
    """
    res = coef[0]
    for index in range(1, num + 1):
        mul_res = np.multiply(res, data_x)
        res = np.add(mul_res, coef[index])
    return res


def _generic_spence_interval(x):
    """
    x < 2, y = x
    x >= 2, y = 1 / x

    y > 1.5, w = 1 / y - 1
    0.5 <= y <= 1.5, w = y - 1
    y < 0.5, w = -y

    output_y = -w * polevel(w, A, 7) / polevel(w, B, 7)
    y < 0.5, output_y = π * π / 6.0 - log(y) * log(1.0 - y) - output_y
    x > 1.5, output_y = - 0.5 * (log(y))^2 - output_y
    Parameters:
    ----------
    x: the placeholder of data input

    output_y : the dict of output

    Returns : A Tensor. Has the same type as data_input.
    -------
    """
    data_rec = np.true_divide(1.0, x)
    if x < 2:
        y = x
    else:
        y = data_rec

    y_rec = np.true_divide(1.0, y)

    if y > 1.5:
        w = np.subtract(y_rec, 1.0)
    elif y < 0.5:
        w = np.multiply(y, -1.0)
    else:
        w = np.subtract(y, 1.0)

    data_a_polevl = polevl(w, Constant.SCALAR_A, Constant.A_COUNT)
    data_b_polevl = polevl(w, Constant.SCALAR_B, Constant.B_COUNT)

    spence_pol = np.multiply(w, data_a_polevl)
    spence_pol = np.true_divide(spence_pol, data_b_polevl)
    spence = np.multiply(spence_pol, -1.0)

    z = np.log(y)
    if y < 0.5:
        m = np.subtract(1, y)
        n = np.log(m)
        spence_y_lt_half = np.multiply(z, n)
        spence_y_lt_half = np.subtract(Constant.SCALAR_PI206, spence_y_lt_half)
        spence = np.subtract(spence_y_lt_half, spence)
    if x > 1.5:
        sqare_z = np.multiply(z, z)
        spence_x_gt_three_half = np.multiply(sqare_z, -0.5)
        spence = np.subtract(spence_x_gt_three_half, spence)
    spence = spence.astype(x.dtype)

    return spence


def calc_expect_func(x, y):
    """
    do element-wise spence compute
    x < 0, y = nan
    x = 0, y = π * π / 6.0
    x = 1, y = 0
    else, y = _generic_spence_interval(x)
    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "spence"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """
    input_x = x["value"]
    for i in np.nditer(input_x, op_flags=['readwrite']):
        if i < 0:
            res = np.nan
        elif i == 0:
            res = Constant.SCALAR_PI206
        elif i == 1:
            res = 0
        else:
            res = _generic_spence_interval(i)
        i[...] = res
    return [input_x, ]
