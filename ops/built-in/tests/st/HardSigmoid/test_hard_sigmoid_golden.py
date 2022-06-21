#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
golden data generation function for HardSigmoid op
'''
import torch
from torch import nn
import numpy as np

def calc_expect_func(input_x, output_y, alpha=0.16666666, beta=0.5):
    m = nn.Hardsigmoid()
    if input_x["dtype"] == 'float16':
        x = torch.tensor(input_x["value"], dtype=torch.float32)
        res = m(x)
        res = res.numpy()
        res = res.astype(np.float16)
        return [res, ]
    if input_x["dtype"] == 'int32':
        x = torch.tensor(input_x["value"], dtype=torch.float32)
        res = m(x)
        res = res.numpy()
        res = res.astype(np.int32)
        return [res, ]
    x = torch.tensor(input_x["value"])
    res = m(x)
    res = res.numpy()
    return [res, ]