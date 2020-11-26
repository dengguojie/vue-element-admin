import numpy as np
from op_test_frame.utils import calc_shape_size


def np_add(x1, x2, y):
    x1_shape = x1["shape"]
    x1_format = x1["format"]
    x1_data = x1["value"][:calc_shape_size(x1_shape)].reshape(x1_shape)
    x2_format = x2["format"]
    x2_shape = x2["shape"]
    x2_data = x2["value"][:calc_shape_size(x2_shape)].reshape(x2_shape)
    if x1_format == x2_format:
        return x1_data + x2_data
    if x1_format == "NZ" and x2_format == "ND":
        x2_new_shape = x2_shape[:-3] + [(x2_shape[-1] + 15) // 16, (x2_shape[-2] + 15) // 16, 1 if x2_shape[-2] == 1 else 16, 1 if x2_shape[-1] == 1 else 16]
        x2_data.reshape(x2_new_shape)

    if x1_format == "ND" and x2_format == "NZ":
        x1_new_shape = x1_shape[:-3] + [(x1_shape[-1] + 15) // 16, (x1_shape[-2] + 15) // 16, 1 if x1_shape[-2] == 1 else 16, 1 if x1_shape[-1] == 1 else 16]
        x1_data.reshape(x1_new_shape)

    return x1_data + x2_data