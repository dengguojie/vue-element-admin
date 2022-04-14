#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for fast tanh
'''
# Third-Party Packages
import numpy as np

def calc_expect_func(input_x, output_y):
    x_value = input_x.get("value")

    dtype = input_x["dtype"]
    if dtype == "fp16" or dtype == "float16":
        s_type = np.float16
    elif dtype == "fp32" or dtype == "float32":
        s_type = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    output = np.tanh(x_value).astype(s_type)
    return output