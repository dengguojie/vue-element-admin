#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
test_accumulate_nv2
'''
import numpy as np


def calc_expect_func(x0, x1, y, N):
    res = x0["value"] + x1["value"]
 
    return [res, ]