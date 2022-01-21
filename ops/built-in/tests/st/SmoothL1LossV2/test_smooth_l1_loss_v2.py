
#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
test_smooth_l1_loss_v2
'''
import torch

def calc_expect_func(predict, label, loss, sigma, reduction):
    predict = torch.tensor(predict["value"])
    label = torch.tensor(label["value"])

    res = torch.nn.functional.smooth_l1_loss(predict, label, reduction="none").numpy()

    return [res, ]
