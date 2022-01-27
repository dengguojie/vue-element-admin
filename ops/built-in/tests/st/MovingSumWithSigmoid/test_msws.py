
#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
test_msws
'''
import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def calc_expect_func(alpha, energy, beam_size, frame_size, y, window_size):
    alpha_array = alpha["value"]
    energy_array = energy["value"]
    beam_size_array = beam_size["value"]
    frame_size_array = frame_size["value"]

    out_shape = energy["shape"]
    res = np.zeros(out_shape)

    batch_size = frame_size["shape"][0]
    col_offset = np.sum(frame_size_array)
 
    alpha_offset_tmp = 0
    enery_row_offset_tmp = 0
    enery_col_offset_tmp = 0

    alpha_offset = [0] * batch_size
    energy_offset = [0] * batch_size

    for b in range(batch_size - 1):
        alpha_offset_tmp += beam_size_array[b] * frame_size_array[b]
        enery_row_offset_tmp += beam_size_array[b]
        enery_col_offset_tmp += frame_size_array[b]
        energ_offset_tmp = enery_row_offset_tmp * col_offset + enery_col_offset_tmp

        alpha_offset[b + 1] = alpha_offset_tmp
        energy_offset[b + 1] = energ_offset_tmp

    for b in range(batch_size):
        frame_num = frame_size_array[b]
        for i in range(frame_num):
            loop = min(frame_num - i, window_size)
            moving_sum = 0.0
            for j in range(loop):
                moving_sum += alpha_array[alpha_offset[b] + i + j]

            res[energy_offset[b] + i] = moving_sum * sigmoid(energy_array[energy_offset[b] + i])

    return [res, ]
