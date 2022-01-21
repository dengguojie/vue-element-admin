
#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
test_msws
'''
import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def calc_expect_func(alpha, energy, frame_size, y, window_size):
    alpha_array = alpha["value"]
    energy_array = energy["value"]
    frame_size_array = frame_size["value"]
    out_shape = alpha["shape"]
    res = np.zeros(out_shape)
    frame_num = frame_size_array[0]
    for i in range(frame_num):
        loop = min(frame_num - i, window_size)
        moving_sum = 0.0
        for j in range(loop):
            moving_sum += alpha_array[i + j]

        res[i] = moving_sum * sigmoid(energy_array[i])

    return [res, ]
