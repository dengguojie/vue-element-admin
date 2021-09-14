import numpy as np
from functools import reduce


def calc_expect_10(x, perm, y):
    print("x=", x)
    print("perm=", perm)
    print("y=", y)
    x_value = x.get("value")
    res = np.transpose(x_value, axes=(1, 0))
    return res 

