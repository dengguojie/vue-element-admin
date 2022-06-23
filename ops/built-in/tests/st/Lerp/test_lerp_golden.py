#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
Special golden data generation function for ops lerp
'''
# Third-Party Packages
import torch
import numpy as np

def calc_expect_func(start, end, weight, y):
    if y["dtype"] == "float16":
        start = torch.tensor(start["value"], dtype=torch.float32)
        end = torch.tensor(end["value"], dtype=torch.float32)
        weight = torch.tensor(weight["value"], dtype=torch.float32)
        res = torch.lerp(start, end, weight)
        res = res.numpy().astype(np.float16)
        return [res, ]
    start = torch.tensor(start["value"])
    end = torch.tensor(end["value"])
    weight = torch.tensor(weight["value"])
    res = torch.lerp(start, end, weight)
    res = res.numpy()
    return [res, ]