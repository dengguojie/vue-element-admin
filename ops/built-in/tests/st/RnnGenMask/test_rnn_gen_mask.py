 
#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
'''
test_rnn_gen_mask
'''
import numpy as np


def calc_expect_func(seq_length, seq_mask, num_step, hidden_size):
    seq_length = seq_length["value"]
    batch_size = len(seq_length)
    one_part = [1] * hidden_size
    zero_part = [0] * hidden_size
    res = []
    for i in range(num_step):
        for val in seq_length:
            if val > i:
                res.append(one_part)
            else:
                res.append(zero_part)
    res = np.array(res)
    res = res.reshape((num_step, batch_size, hidden_size)).astype("float16")
    return [res, ]
