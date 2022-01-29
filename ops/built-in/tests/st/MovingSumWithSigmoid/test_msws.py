
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
    energy_row_offset = [0] * batch_size
    energy_col_offset = [0] * batch_size

    for b in range(batch_size - 1):
        alpha_offset_tmp += beam_size_array[b] * frame_size_array[b]
        enery_row_offset_tmp += beam_size_array[b]
        enery_col_offset_tmp += frame_size_array[b]
 
        alpha_offset[b + 1] = alpha_offset_tmp
        energy_row_offset[b + 1] = enery_row_offset_tmp
        energy_col_offset[b + 1] = enery_col_offset_tmp

    for b in range(batch_size):
        beam_size = beam_size_array[b]
        frame_num = frame_size_array[b]
        alpha_offset_v = alpha_offset[b]
        energy_row_offset_v = energy_row_offset[b]
        energy_col_offset_v = energy_col_offset[b]
        for beam_idx in range(beam_size):
            for i in range(frame_num):
                loop = min(frame_num - i, window_size)
                moving_sum = 0.0
                for j in range(loop):
                    moving_sum += alpha_array[alpha_offset_v + beam_idx * frame_num + i + j]

                res[(energy_row_offset_v + beam_idx) * col_offset + energy_col_offset_v + i] = moving_sum *
                    sigmoid(energy_array[(energy_row_offset_v + beam_idx) * col_offset + energy_col_offset_v + i]

    return [res, ]
