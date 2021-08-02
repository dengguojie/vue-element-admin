#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_add_golden
"""

'''
Special golden data generation function for ops add
'''

def calc_expect_func(x1, x2, y1):
    res = x1["value"] + x2["value"]
    return [res,  ]