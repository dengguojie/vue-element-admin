import os
import numpy as np

def calc_expect_func(input_x1, input_x2, output):
    res = np.less(input_x1["value"], input_x2["value"])
    return [res,]