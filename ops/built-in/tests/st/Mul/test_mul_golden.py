#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_mul_golden
"""

'''
Special golden data generation function for ops mul
'''

def calc_expect_func(x1, x2, y):
    res = x1["value"] * x2["value"]
    return [res]
