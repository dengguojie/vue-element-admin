import sys

from numpy.core.shape_base import block
from op_test_frame.ut import BroadcastOpUT
import numpy as np
import pprint
from op_test_frame.common import precision_info

def depth_to_space_forward(x, y, block_size, mode='DCR', data_format='NHWC', kernel_name='depth_to_space'):
    y_dtype = y.get('dtype')
    input_data = x.get('value')
    if mode == 'DCR' and data_format == 'NCHW':
        b, c, h, w = x.get('shape')
        tmp = np.reshape(input_data, [b, block_size, block_size, c//(block_size**2), h, w])
        tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
        res = np.reshape(tmp, [b, c//(block_size ** 2), h * block_size, w * block_size])
    elif mode == 'CRD' and data_format == 'NCHW':
        b, c, h, w = x.get('shape')
        tmp = np.reshape(input_data, [b, c//(block_size**2), block_size, block_size, h, w])
        tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
        res = np.reshape(tmp, [b, c//(block_size ** 2), h * block_size, w * block_size])
    elif mode == 'DCR' and data_format == 'NHWC':
        b, h, w, c = x.get('shape')
        tmp = np.reshape(input_data, [b, h, w, block_size, block_size, c//(block_size**2)])
        tmp = np.transpose(tmp, [0, 1, 3, 2, 4, 5])
        res = np.reshape(tmp, [b, h * block_size, w * block_size, c//(block_size ** 2)])
    else :
        b, h, w, c = x.get('shape')
        tmp = np.reshape(input_data, [b, h, w, c//(block_size**2), block_size, block_size])
        tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
        res = np.reshape(tmp, [b, h * block_size, w * block_size, c//(block_size ** 2)])
    res = res.astype(y_dtype)
    return res