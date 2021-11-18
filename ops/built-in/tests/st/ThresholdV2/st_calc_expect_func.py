import numpy as np

# 'pylint: disable=unused-argument
def calc_expect_func(x, threshold, y):
    input_x = x["value"]
    threshold_val = threshold["value"][0]

    for i in np.nditer(input_x, op_flags=['readwrite']):
        if i <= threshold_val:
            i[...] = 0
    return [input_x, ]