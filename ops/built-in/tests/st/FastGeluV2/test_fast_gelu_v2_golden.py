#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for fast gelu v2
'''
# Third-Party Packages
import numpy as np


def calc_expect_func(input_x, output_y, kernel_name="fast_gelu_v2", impl_mode=None):
    x = input_x['value']
    const_a = -1.769
    const_b = 1.769
    const_a_half = -0.1444  # a/2
    const_c = 0.7071  # 1/sqrt(2)
    const_offset = 0.000000000001
    const_d = 0.5

    mul_x =  np.array(x) * const_c
    abs_muls = np.abs(mul_x)
    max_abs_muls = np.minimum(abs_muls, const_b)
    vadds = max_abs_muls + const_a
    temp = np.square(vadds)
    temp_0 = temp * const_a_half
    temp_0 = np.add(temp_0, const_d)
    x_adds = np.add(x, const_offset)
    x_abs = np.abs(x_adds)
    sgn = x_adds / x_abs
    temp_1 = temp_0 * sgn
    temp_1 = temp_1 + const_d
    res = x * temp_1
    return res
