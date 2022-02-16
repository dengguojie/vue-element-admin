import numpy as np


def calc_expect_func_case_3(x, y0, y1, y2, y3, size_splits, split_dim, num_split):
    result = np.split(x["value"], 4, 1)
    return result
