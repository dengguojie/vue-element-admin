#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
Special golden data generation function for ops cdist
test_cdist_golden
'''
import numpy as np

def calc_expect_func(x1, x2, y, p=2.0):
    dtype = x1["dtype"]
    if dtype == "float16":
        x1_data = x1["value"].astype(np.float32)
        x2_data = x2["value"].astype(np.float32)
    else:
        x1_data = x1["value"]
        x2_data = x2["value"]

    if p == 0.0:
        elements = (x1_data != x2_data)
        res = np.sum(elements, -1)
    else:
        diff = np.subtract(x1_data, x2_data)
        diff = np.abs(diff)

        if p == np.inf:
            res = np.max(diff, -1)
        elif p == 1:
            res = np.sum(diff, -1)
        else:
            elements = diff ** p
            summation = np.sum(elements, -1)
            res = summation ** (1/p)
    if dtype == "float16":
        ret = res.astype(np.float16)
    else:
        ret = res
    return [ret, ]