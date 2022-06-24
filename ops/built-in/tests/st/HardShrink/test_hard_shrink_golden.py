#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import numpy as np


def calc_expect_func(input_x, output_y, lambd):
    
    if output_y["dtype"] == 'float16':
        input_x = torch.tensor(input_x["value"], dtype=torch.float32)
        res = torch.nn.functional.hardshrink(input_x, lambd)
        res = res.numpy()
        res = res.astype(np.float16)
        return [res, ]

    input_x = torch.tensor(input_x["value"])
    res = torch.nn.functional.hardshrink(input_x, lambd)
    res = res.numpy()
 
    return [res, ]