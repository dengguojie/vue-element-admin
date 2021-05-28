import os
import numpy as np

def get_data_type(dst_type):
    if dst_type == 0:
        return np.float32
    elif dst_type == 1:
        return np.float16
    elif dst_type == 2:
        return np.int8
    elif dst_type == 3:
        return np.int32
    elif dst_type == 4:
        return np.uint8
    elif dst_type == 6:
        return np.int16
    elif dst_type == 7:
        return np.uint16
    elif dst_type == 8:
        return np.uint32
    elif dst_type == 9:
        return np.int64
    elif dst_type == 10:
        return np.uint64
    elif dst_type == 11:
        return np.float64
    elif dst_type == 12:
        return np.bool
    elif dst_type == 16:
        return np.complex64
    elif dst_type == 17:
        return np.complex128


def calc_expect_func(input_x, input_y, dst_type):
    res = input_x["value"].astype(get_data_type(dst_type))
    return [res,]