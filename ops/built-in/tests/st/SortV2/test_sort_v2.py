
#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
test_sort_v2
'''
import torch
import numpy as np

def calc_expect_func(x, y, axis=-1, descending=False):
    torch_x = torch.tensor(x["value"], dtype=torch.float32)

    y, _ = torch.sort(torch_x, dim=axis, descending=descending)
    y = y.numpy()
    if x["dtype"] == "float16":
        y = y.astype(np.float16)

    return [y, ]
