#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for einusm pattern
'''
# Third-Party Packages
import numpy as np

def calc_expect_func(x1, x2, y, equation):
    x1 = x1.get('value').astype(np.float32)
    x2 = x2.get('value').astype(np.float32)

    # np einsum
    res_nd = np.einsum(equation, x1, x2).astype(np.float16)
    return res_nd

