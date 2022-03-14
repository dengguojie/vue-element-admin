# Copyright 2022 Huawei Technologies Co., Ltd
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
st for Expint
"""
import numpy as np


EUL = 5.772156649015328606065e-1
list_a = np.array([-5.350447357812542947283E0,
        2.185049168816613393830E2,
        -4.176572384826693777058E3,
        5.541176756393557601232E4,
        -3.313381331178144034309E5,
        1.592627163384945414220E6], dtype='float32')
list_b = np.array([-5.250547959112862969197E1,
        1.259616186786790571525E3,
        -1.756549581973534652631E4,
        1.493062117002725991967E5,
        -7.294949239640527645655E5,
        1.592627163384945429726E6], dtype='float32')
list_a2 = np.array([-2.106934601691916512584E0,
        1.732733869664688041885E0,
        -2.423619178935841904839E-1,
        2.322724180937565842585E-2,
        2.372880440493179832059E-4,
        -8.343219561192552752335E-5,
        1.363408795605250394881E-5,
        -3.655412321999253963714E-7,
        1.464941733975961318456E-8,
        6.176407863710360207074E-10], dtype='float32')
list_b2 = np.array([-2.298062239901678075778E-1,
        1.105077041474037862347E-1,
        -1.566542966630792353556E-2,
        2.761106850817352773874E-3,
        -2.089148012284048449115E-4,
        1.708528938807675304186E-5,
        -4.459311796356686423199E-7,
        1.394634930353847498145E-8,
        6.150865933977338354138E-10], dtype='float32')
list_a3 = np.array([-7.657847078286127362028E-1,
        6.886192415566705051750E-1,
        -2.132598113545206124553E-1,
        3.346107552384193813594E-2,
        -3.076541477344756050249E-3,
         1.747119316454907477380E-4,
        -6.103711682274170530369E-6,
        1.218032765428652199087E-7,
        -1.086076102793290233007E-9], dtype='float32')
list_b3 = np.array([-1.888802868662308731041E0,
        1.066691687211408896850E0,
        -2.751915982306380647738E-1,
        3.930852688233823569726E-2,
        -3.414684558602365085394E-3,
        1.866844370703555398195E-4,
        -6.345146083130515357861E-6,
        1.239754287483206878024E-7,
        -1.086076102793126632978E-9], dtype='float32')
list_a4 = np.array([-2.458119367674020323359E-1,
        -1.483382253322077687183E-1,
        7.248291795735551591813E-2,
        -1.348315687380940523823E-2,
        1.342775069788636972294E-3,
        -7.942465637159712264564E-5,
        2.644179518984235952241E-6,
        -4.239473659313765177195E-8], dtype='float32')
list_b4 = np.array([-1.044225908443871106315E-1,
        -2.676453128101402655055E-1,
        9.695000254621984627876E-2,
        -1.601745692712991078208E-2,
        1.496414899205908021882E-3,
        -8.462452563778485013756E-5,
        2.728938403476726394024E-6,
        -4.239462431819542051337E-8], dtype='float32')
list_a5 = np.array([-1.373215375871208729803E0,
        -7.084559133740838761406E-1,
        1.580806855547941010501E0,
        -2.601500427425622944234E-1,
        2.994674694113713763365E-2,
        -1.038086040188744005513E-3,
        4.371064420753005429514E-5,
        2.141783679522602903795E-6], dtype='float32')
list_b5 = np.array([8.585231423622028380768E-1,
        4.483285822873995129957E-1,
        7.687932158124475434091E-2,
        2.449868241021887685904E-2,
        8.832165941927796567926E-4,
        4.590952299511353531215E-4,
        -4.729848351866523044863E-6,
        2.665195537390710170105E-6], dtype='float32')
list_a6 = np.array([1.981808503259689673238E-2,
        -1.271645625984917501326E0,
        -2.088160335681228318920E0,
        2.755544509187936721172E0,
        -4.409507048701600257171E-1,
        4.665623805935891391017E-2,
        -1.545042679673485262580E-3,
        7.059980605299617478514E-5], dtype='float32')
list_b6 = np.array([1.476498670914921440652E0,
        5.629177174822436244827E-1,
        1.699017897879307263248E-1,
        2.291647179034212017463E-2,
        4.450150439728752875043E-3,
        1.727439612206521482874E-4,
        3.953167195549672482304E-5], dtype='float32')
list_a7 = np.array([1.212561118105456670844E-1,
        -5.823133179043894485122E-1,
        2.348887314557016779211E-1,
        -3.040034318113248237280E-2,
        1.510082146865190661777E-3,
        -2.523137095499571377122E-5], dtype='float32')
list_b7 = np.array([-1.002252150365854016662E0,
        2.928709694872224144953E-1,
        -3.337004338674007801307E-2,
        1.560544881127388842819E-3,
        -2.523137093603234562648E-5], dtype='float32')


def polevl_plevl(inp_x, ans, bns, iter_n, iter_m):
    """
    expint
                 1       2       6
    y = 1 + ---  +  ---  +  ---- + ...
             x        2       3
                     x       x
    """
    res = ans[0]
    for i in range(1, iter_n+1):
        mul_res = np.multiply(res, inp_x)
        res = np.add(mul_res, ans[i])
    data_a = res
    _res = np.add(bns[0], inp_x)
    for i in range(1, iter_m+1):
        mul_res = np.multiply(_res, inp_x)
        _res = np.add(mul_res, bns[i])
    data_b = _res
    data_f = np.true_divide(data_a,data_b)

    return data_f


def expint_cal(inp_x, ans, bns, iter_n, iter_m):
    """
    expint
    """
    data_w = np.true_divide(1.0, inp_x)
    _res = polevl_plevl(data_w, ans, bns, iter_n, iter_m)
    mul_res = np.multiply(data_w, _res)
    add_res = np.add(mul_res, 1.0)
    mul_val = np.multiply(add_res, data_w)
    exp_val = np.exp(inp_x)
    res = np.multiply(mul_val, exp_val)

    return res


def calc_expect_func(x,y):
    """
    calc_expect_func
    """
    input_x = x["value"]
    for i in np.nditer(input_x, op_flags=['readwrite']):
        if i <= 0.0:
            i[...] = 0.0
        elif i < 2.0:
            res = polevl_plevl(i, list_a, list_b, 5, 5)
            mul_res = np.multiply(res, i)
            log_res = np.log(i)
            add_res = np.add(mul_res, log_res)
            i[...] = np.add(EUL, add_res)
        elif i < 4.0:
            i[...] = expint_cal(i, list_a6, list_b6, 7, 6)
        elif i < 8.0:
            i[...] = expint_cal(i, list_a5, list_b5, 7, 7)
        elif i < 16.0:
            i[...] = expint_cal(i, list_a2, list_b2, 9, 8)
        elif i < 32.0:
            i[...] = expint_cal(i, list_a4, list_b4, 7, 7)
        elif i < 64.0:
            i[...] = expint_cal(i, list_a7, list_b7, 5, 4)
        else:
            i[...] = expint_cal(i, list_a3, list_b3, 8, 8)

    return [input_x, ]
