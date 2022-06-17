#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
golden data generation function for Swish op
'''
import torch
import numpy as np

def calc_expect_func(x, y, scale):
    if x["dtype"] == 'float16':
        x = torch.tensor(x["value"], dtype=torch.float32)
        res = x * torch.sigmoid(x)
        res = res.numpy()
        res = res.astype(np.float16)
        return [res, ]
    x = torch.tensor(x["value"])
    res = x * torch.sigmoid(x)
    res = res.numpy()
    return [res, ]