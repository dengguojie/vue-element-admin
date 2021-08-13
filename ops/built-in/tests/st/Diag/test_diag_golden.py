#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import numpy as np


def calc_expect_func(x, y):
    res = np.diag(x["value"])
    return [res,]
