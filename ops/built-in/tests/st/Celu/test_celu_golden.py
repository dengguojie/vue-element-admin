#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
Special golden data generation function for ops celu
test_celu_golden
'''
import torch

def calc_expect_func(input_x, output_y, alpha=1.0):
    x = input_x["value"]
    torch_x = torch.Tensor(x)
    
    res = torch.celu(torch_x, alpha=alpha).numpy()
    return [res, ]
