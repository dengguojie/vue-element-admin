#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
Special golden data generation function for ops cdist
test_cdist_grad_golden
'''

import torch
import numpy as np

#pylint: disable=unused-argument
def calc_expect_func(input_grad, input_x1, input_x2, input_cdist, y, p):
    x1_value = torch.tensor(input_x1["value"], dtype=torch.float32,
                            requires_grad=True)
    x2_value = torch.tensor(input_x2["value"], dtype=torch.float32,
                            requires_grad=True)
    grad_value = torch.tensor(input_grad["value"], dtype=torch.float32)

    cdist_value = torch.tensor(input_cdist["value"], dtype=torch.float32)

    res = cdist_backward(x1_value, x2_value, p, grad_value, cdist_value)
    if input_x1["dtype"] == "float16":
        res = res.half()
    return [res.numpy(), ]


def cdist_backward(x1, x2, p, grad, cdist):
    diff = x1 - x2
    diff_abs = torch.abs(diff)
    nz_cdist = torch.where(cdist == 0, torch.ones_like(cdist), cdist)
    sign = torch.where(diff > 0, torch.ones_like(diff),
                       torch.full_like(diff, -1))
    sign = torch.where(diff == 0, torch.zeros_like(diff), sign)

    if p == 0.0:
        res = torch.zeros_like(diff)
    elif p == 1.0:
        res = grad * sign
    elif p < 2.0:
        res = sign * torch.pow(diff_abs, p - 1.0) * grad / torch.pow(nz_cdist,
                                                                     p - 1.0)
        res = torch.where(cdist == 0, torch.zeros_like(res), res)
    elif p == 2.0:
        res = grad * diff / nz_cdist
        res = torch.where(cdist == 0, torch.zeros_like(res), res)
    elif p == float("inf"):
        mask = torch.where(cdist - diff_abs > 0, torch.zeros_like(diff),
                           torch.ones_like(diff))
        res = grad * sign * mask
    else:
        res = diff * torch.pow(diff_abs, p - 2) * grad / torch.pow(nz_cdist,
                                                                   p - 1.0)
        res = torch.where(cdist == 0, torch.zeros_like(res), res)
    res = torch.sum(res, -2)
    return res.detach()