#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_logsigmoid_golden
"""

import torch
import numpy as np


def calc_expect_func(x, y):
    in_dtype = x["dtype"]
    input_data = x["value"].astype(np.float32)
    tensor_data = torch.from_numpy(input_data)
    m = torch.nn.LogSigmoid()
    res = m(tensor_data)
    res = res.numpy()
    if in_dtype == "float16":
        res = res.astype(in_dtype)
    return [res, ]
