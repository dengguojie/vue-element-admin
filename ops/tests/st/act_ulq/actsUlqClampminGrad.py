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
import mindspore
import mindspore.context as context
context.set_context(save_graphs=True)
RESOLUTION = np.finfo(np.float64).resolution
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
def ActsULQClampMinGrad(y_grad, clamp_min_mask, x_clamped_loss):
    signal = np.where(clamp_min_mask, 0, 1).astype(y_grad.dtype)
    x_min_grad = signal - x_clamped_loss
    clamp_min_grad = np.sum(y_grad * x_min_grad)
    return [clamp_min_grad]

def ActsULQClampMinGradTBE(y_grad, clamp_min_mask, x_clamp_loss):
    x_grad  = P.ActULQClampMinGrad()(Tensor(y_grad), Tensor(clamp_min_mask, mindspore.bool_), Tensor(x_clamp_loss))

    return x_grad.asnumpy()

def TestULQ(y_grad, clamp_min_mask, x_clamp_loss):
    numpy_result =ActsULQClampMinGrad(y_grad, clamp_min_mask, x_clamp_loss)
    tbe_result = ActsULQClampMinGradTBE(y_grad, clamp_min_mask, x_clamp_loss)
    print(clamp_min_mask)
    print(x_clamp_loss)
    print(numpy_result)
    print(tbe_result)
    for i in range(len(numpy_result)):
        print(calculate_cosine_similarity(numpy_result[i],tbe_result[i]))


if __name__ == '__main__':
    os.system('rm kernel_meta/ -rf')
    data_type = np.float32
    shape = (2,3,4)
    x= np.random.uniform(-10, 10, shape).astype(data_type)
    flat= np.full(shape, False)
    flat = flat.flatten()
    flat[:flat.shape[0]//2] = True
    np.random.shuffle(flat)
    clamp_min_mask = flat.reshape(shape)
    x_clamp_loss = np.random.uniform(-10, 10, shape).astype(data_type)
    TestULQ(x, clamp_min_mask, x_clamp_loss)

