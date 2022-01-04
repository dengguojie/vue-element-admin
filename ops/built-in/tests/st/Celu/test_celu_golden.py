#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
Special golden data generation function for ops celu
test_celu_golden
'''
import torch

def calc_expect_func(x, y, alpha=1.0):
    x_array = x["value"]
    torch_x = torch.Tensor(x_array)
    
    res = torch.celu(torch_x, alpha=alpha).numpy()
    return [res, ]
