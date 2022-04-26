#!/usr/bin/env python3
# _*_ coding: UTF-8 _*_
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import numpy as np


def broadcast_to(x, shape, y, kernel_name="broadcast_to"):
    x_dtype = x.get('dtype')
    shape = list(shape.get('value'))
    input_data = x.get('value')
    res = np.broadcast_to(input_data, shape)
    res = res.astype(x_dtype)
    return res
