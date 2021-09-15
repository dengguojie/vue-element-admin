import sys

import numpy as np

def space_to_depth_forward(x, y, block_size, data_format="NHWC"):
    x_value = x.get('value')
    batch, height, width, depth = x.get('shape')
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = np.reshape(x_value, [batch, reduced_height, block_size, reduced_width, block_size, depth])
    z = np.transpose(y, [0, 1, 3, 2, 4, 5])
    res = np.reshape(z, [batch, reduced_height, reduced_width, -1])
    return res
