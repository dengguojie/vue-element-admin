#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
Special golden data generation function for ops celu
test_celu_golden
'''
import torch
import numpy as np

def calc_expect_func(x, y, alpha=1.0):
    indtype = x["dtype"]
    x = x["value"].astype(np.float32)
    torch_x = torch.tensor(x)
    res = torch.celu(torch_x, alpha=alpha).numpy()
    if indtype == "float16":
        res = res.astype(indtype)
    return [res, ]
