#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_trace_golden
"""

import torch
import numpy

# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
def calc_expect_func(x, y):
    torch_x = torch.tensor(x["value"], dtype=torch.float32)
    res = torch.trace(torch_x)
    if x["dtype"] == "float16":
        res = res.half()
    return [res.numpy(), ]