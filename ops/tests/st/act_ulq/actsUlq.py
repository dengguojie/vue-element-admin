"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

test ifmr
"""


import os
import shutil
import unittest
import numpy as np

from mindspore.ops import operations as P
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn.cell import Cell

RESOLUTION = np.finfo(np.float64).resolution
EPS = 1.192092896e-07
def calculate_cosine_similarity(a, b):
    a = a.astype(np.float64) + RESOLUTION
    b = b.astype(np.float64) + RESOLUTION
    if a.shape != b.shape:
        raise RuntimeError('a shape {} != b shape {}'.format(a.shape, b.shape))
    if a.max() == a.min() and b.max() == b.min():
        return 1
    numerator = np.sum(a * b)
    denominator = ((np.sum(a * a)) ** 0.5) * ((np.sum(b * b)) ** 0.5)
    return numerator / denominator

def ActsULQ(x, clamp_min, clamp_max, fixed_min, num_bits):
    N = np.power(2, num_bits) - 1
    N = N.astype(x.dtype)
    if fixed_min or clamp_min > 0:
        clamp_min = np.array(0, x.dtype)
    if clamp_max < N*EPS:
        clamp_max = np.array(N*EPS, x.dtype)
    scale = (clamp_max - clamp_min) / N
    offset = np.round(clamp_min / scale)
    clamp_min_loss = np.round(clamp_min/scale) - clamp_min/scale
    clamp_min_loss = clamp_min_loss / N
    clamp_min = scale * offset
    clamp_max = scale * (offset + N)
    clamp_min_mask = x >= clamp_min
    clamp_max_mask = x <= clamp_max
    x = np.clip(x, clamp_min, clamp_max, dtype=x.dtype)
    quant_x = scale * np.round(x / scale)
    x_clamped_loss = (np.round(x / scale) - x / scale ) / N
    x_clamped_loss = np.where(clamp_min_mask, x_clamped_loss, clamp_min_loss)
    x_clamped_loss = np.where(clamp_max_mask, x_clamped_loss, clamp_min_loss)
    return quant_x, clamp_min_mask, clamp_max_mask, x_clamped_loss

def ActsULQTBE(x, clamp_min, clamp_max, fixed_min, num_bits):
    quant_x, clamp_min_mask, clamp_max_mask, x_clamped_loss = P.ActsULQ(fixed_min, num_bits)(Tensor(x), Tensor( clamp_min), Tensor(clamp_max))

    return quant_x.asnumpy(), clamp_min_mask.asnumpy(), clamp_max_mask.asnumpy(), x_clamped_loss.asnumpy()

def TestULQ(x, clamp_min, clamp_max, fixed_min, num_bits):
    numpy_result =ActsULQ(x, clamp_min, clamp_max, fixed_min, num_bits)
    tbe_result = ActsULQTBE(x, clamp_min, clamp_max, fixed_min, num_bits)
    print(x)
    print(clamp_min)
    print(clamp_max)
    print('-------------------------------------------')
    print(numpy_result)
    print(tbe_result)
    for i in range(len(numpy_result)):
        print(calculate_cosine_similarity(numpy_result[i],tbe_result[i]))


if __name__ == '__main__':
    os.system('rm kernel_meta/ -rf')
    data_type = np.float32
    x= np.random.uniform(-10, 10, (32, 120)).astype(data_type)
    clamp_max = 0.7 * np.max(x)
    clamp_min = 0.7 * np.min(x)
    clamp_max = np.array([clamp_max], dtype=data_type)
    clamp_min = np.array([clamp_min], dtype=data_type)
    TestULQ(x, clamp_min, clamp_max,False, 8)
    data_type = np.float16
    x= np.random.uniform(-1, 1, (32, 120)).astype(data_type)
    clamp_max = 0.7 * np.max(x)
    clamp_min = 0.7 * np.min(x)
    clamp_max = np.array([clamp_max], dtype=data_type)
    clamp_min = np.array([clamp_min], dtype=data_type)
    TestULQ(x, clamp_min, clamp_max, False, 8)
