import numpy as np


def calc_sizeof(x):
    dtype = x.get("dtype")
    if dtype == "int64" or dtype == "uint64":
        return 8
    if dtype == "int32" or dtype == "uint32"  or dtype == "float32":
        return 4
    if dtype == "int16" or dtype == "uint16"  or dtype == "float16":
        return 2
    if dtype == "int8":
        return 1
    return 0

def calc_expect_func(x, size, stride, storage_offset, y):
    expect = np.lib.stride_tricks.as_strided(x.get("value"), size.get("value"), stride.get("value") * calc_sizeof(x))
    return expect
