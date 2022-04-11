# Copyright (c) Huawei Technologies Co., Ltd.2022. All rights reserved.
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
'''
test_xlog1py_golden
'''

import numpy as np

SQRTH = 0.70710678118654752440
SQRT2 = 1.41421356237309504880

LP = [
 4.5270000862445199635215E-5,
 4.9854102823193375972212E-1,
 6.5787325942061044846969E0,
 2.9911919328553073277375E1,
 6.0949667980987787057556E1,
 5.7112963590585538103336E1,
 2.0039553499201281259648E1,
]
LQ = [
 1.5062909083469192043167E1,
 8.3047565967967209469434E1,
 2.2176239823732856465394E2,
 3.0909872225312059774938E2,
 2.1642788614495947685003E2,
 6.0118660497603843919306E1,
]

def polevl(x, coef, n):
    """
    def polevl

    Evaluates polynomial of degree N:

                       2          N
    y  =  C  + C x + C x  +...+ C x
          0    1     2          N

    Coefficients are stored in reverse order:

    coef[0] = C  , ..., coef[N] = C  .
              N                   0

    Parameters:
    ----------
    x : the placeholder of data input
    coef : coef of the data
    n: number of the coef

    Returns : A number of polevl() result
    -------
    """

    p = coef
    ans = 0

    for iter_n in range(0, n+1):
        if iter_n == 0:
            ans = p[iter_n]
        else:
            ans = ans * x + p[iter_n]

    return ans


def p1evl(x, coef, n):
    """
    def p1evl

    The function p1evl() assumes that coef[N] = 1.0 and is
    omitted from the array.  Its calling arguments are
    otherwise the same as polevl().

    Parameters:
    ----------
    x : the placeholder of data input
    coef : coef of the data
    n : number of the coef

    Returns : A number of p1evl() result
    -------
    """

    p = coef
    ans = 0

    for iter_n in range(0, n):
        if iter_n == 0:
            ans = x + p[iter_n]
        else:
            ans = ans * x + p[iter_n]

    return ans


def calc_expect_func(x, y, z):
    """
    calc_expect_func
    """
    out_tensor = y["value"]

    for iter_y in np.nditer(out_tensor, op_flags=['readwrite']):
        y_add1 = iter_y + 1

        if (y_add1 < SQRTH) or (y_add1 > SQRT2):
            iter_y[...] = np.log(y_add1)
        else:
            y_pow2 = iter_y * iter_y
            y_pow2 = -0.5 * y_pow2 + iter_y * (y_pow2 * polevl(iter_y, LP, 6) / p1evl(iter_y, LQ, 6))
            iter_y[...] = iter_y + y_pow2
    res = x["value"] * out_tensor
    return [res, ]

