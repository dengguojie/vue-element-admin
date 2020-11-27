#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import random

import te.platform as tbe_platform
from te.lang.cce.te_compute.dilation_compute import calc_minimum_ub
from te.lang.cce.te_compute.dilation_compute import shape_align


def generate_test_case(num):
    res = []
    ub_size = tbe_platform.get_soc_spec("UB_SIZE")
    for i in range(num):
        dim = random.randint(2, 6)
        shape = []
        dilations = []
        for j in range(0, dim):
            shape.append(random.randint(1, 10))
            dilations.append(random.randint(1, 5))
        dilations[-1] = 1
        dtype_map = ["int8", "float16", "float32"]
        dtype = dtype_map[random.randint(1, 10) % 3]
        min_size = calc_minimum_ub(shape, dilations, dtype)
        if min_size > ub_size:
            continue
        padding_value = round(random.random() * 5, 5)
        shape_aligned_x = shape_align(shape, dtype)
        shape_dilated = list(map(lambda a, b: a * b, shape_aligned_x, dilations))
        output_shape = list(map(lambda a, b: a - b + 1, shape_dilated, dilations))
        output_shape[-1] = shape[-1]

        tmp = [dtype, shape, dilations, padding_value, output_shape]
        res.append(tmp)
    return res
