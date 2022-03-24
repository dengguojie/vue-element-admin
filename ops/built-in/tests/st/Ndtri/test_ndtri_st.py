# Copyright (C) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
st for Ndtri
"""
import numpy as np


S2PI = 2.50662827463100050242E0
LIST_P0 = np.array([
    -5.99633501014107895267E1, 9.80010754185999661536E1, -5.66762857469070293439E1, 1.39312609387279679503E1,
    -1.23916583867381258016E0
],
                   dtype='float32')
LIST_Q0 = np.array([
    1.95448858338141759834E0, 4.67627912898881538453E0, 8.63602421390890590575E1, -2.25462687854119370527E2,
    2.00260212380060660359E2, -8.20372256168333339912E1, 1.59056225126211695515E1, -1.18331621121330003142E0
],
                   dtype='float32')
LIST_P1 = np.array([
    4.05544892305962419923E0, 3.15251094599893866154E1, 5.71628192246421288162E1, 4.40805073893200834700E1,
    1.46849561928858024014E1, 2.18663306850790267539E0, -1.40256079171354495875E-1, -3.50424626827848203418E-2,
    -8.57456785154685413611E-4
],
                   dtype='float32')
LIST_Q1 = np.array([
    1.57799883256466749731E1, 4.53907635128879210584E1, 4.13172038254672030440E1, 1.50425385692907503408E1,
    2.50464946208309415979E0, -1.42182922854787788574E-1, -3.80806407691578277194E-2, -9.33259480895457427372E-4
],
                   dtype='float32')
LIST_P2 = np.array([
    3.23774891776946035970E0, 6.91522889068984211695E0, 3.93881025292474443415E0, 1.33303460815807542389E0,
    2.01485389549179081538E-1, 1.23716634817820021358E-2, 3.01581553508235416007E-4, 2.65806974686737550832E-6,
    6.23974539184983293730E-9
],
                   dtype='float32')
LIST_Q2 = np.array([
    6.02427039364742014255E0, 3.67983563856160859403E0, 1.37702099489081330271E0, 2.16236993594496635890E-1,
    1.34204006088543189037E-2, 3.28014464682127739104E-4, 2.89247864745380683936E-6, 6.79019408009981274425E-9
],
                   dtype='float32')


def polevl_plevl(inp_x, ans, bns, iter_n, iter_m):
    """
    ndtri
                 1       2       6
    y = 1 + ---  +  ---  +  ---- + ...
             x        2       3
                     x       x
    """
    res = ans[0]
    for i in range(1, iter_n + 1):
        mul_res = np.multiply(res, inp_x)
        res = np.add(mul_res, ans[i])
    data_a = res
    _res = np.add(bns[0], inp_x)
    for i in range(1, iter_m + 1):
        mul_res = np.multiply(_res, inp_x)
        _res = np.add(mul_res, bns[i])
    data_b = _res
    data_f = np.true_divide(data_a, data_b)

    return data_f


# 'pylint: disable=too-many-locals
def cal_ndtri(input_x):
    """
    do element-wise ndtri compute
    `y = sqrt(2) * erfinv(2 * x - 1)`
    """
    code = 1
    res_y = input_x
    res_exp = 1.0 - 0.13533528323661269189
    if res_y > res_exp:
        res_y = 1.0 - res_y
        code = 0

    if res_y > 0.13533528323661269189:
        res_y = res_y - 0.5
        res_mul = np.multiply(res_y, res_y)
        pp_val = polevl_plevl(res_mul, LIST_P0, LIST_Q0, 4, 7)
        tmp_pp = np.multiply(res_mul, pp_val)
        tmp_mul = np.multiply(res_y, tmp_pp)
        res_x = tmp_mul + res_y
        res = np.multiply(res_x, S2PI)

        return res

    tmp_log = np.log(res_y)
    tmp_mul = np.multiply(tmp_log, -2.0)
    res_x = np.sqrt(tmp_mul)
    log_x = np.log(res_x)
    div_x = np.true_divide(log_x, res_x)
    res_sub = res_x - div_x

    res_z = np.true_divide(1.0, res_x)

    pp_val1 = polevl_plevl(res_z, LIST_P1, LIST_Q1, 8, 7)
    pp_val2 = polevl_plevl(res_z, LIST_P2, LIST_Q2, 8, 7)
    if res_x < 8.0:
        res_x1 = np.multiply(res_z, pp_val1)
    else:
        res_x1 = np.multiply(res_z, pp_val2)

    res_x = res_sub - res_x1
    if code != 0:
        res_x = -res_x

    return res_x


# 'pylint: disable=unused-argument,invalid-name
def calc_expect_func(x, y):
    """
    calc_expect_func
    """
    input_x = x["value"]
    for i in np.nditer(input_x, op_flags=['readwrite']):
        if i <= -1.0:
            i[...] = 0.0
        elif i < 1.0:
            i[...] = cal_ndtri(i)
        else:
            i[...] = 0.0

    return [
        input_x,
    ]
