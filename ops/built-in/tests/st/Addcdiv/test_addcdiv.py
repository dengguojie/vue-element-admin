#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
test_addcdiv
'''
import torch
import numpy as np

def calc_expect_func(input_data, x1, x2, value, y):
    if y["dtype"] == 'float16':
        input_data = torch.tensor(input_data["value"], dtype=torch.float32)
        x1 = torch.tensor(x1["value"], dtype=torch.float32)
        x2 = torch.tensor(x2["value"], dtype=torch.float32)
        value_tensor = torch.tensor(value["value"], dtype=torch.float32)

        res = torch.addcdiv(input_data, x1, x2, value=value_tensor[0])
        res = res.numpy()
        res = res.astype(np.float16)
        return [res, ]

    input_data = torch.tensor(input_data["value"])
    x1 = torch.tensor(x1["value"])
    x2 = torch.tensor(x2["value"])
    value_tensor = torch.tensor(value["value"])
    
    res = torch.addcdiv(input_data, x1, x2, value=value_tensor[0])
    res = res.numpy()
 
    return [res, ]