#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization


def test_conv2d_backprop_input_fuzz_build_lower_limit():
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (16, 3, 16, 16)
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (-1, 3, -1, -1, 16),
            'ori_shape': (-1, 33, -1, -1),
            'range': ((16, 31), (3, 3), (4, 15), (4, 15), (16, 16)),
            'ori_range': ((16, 31), (33, 33), (4, 15), (4, 15)),
            'ori_format': 'NHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 3, -1, -1),
            'range': ((16, 31), (1, 1), (16, 31), (16, 31), (16, 16)),
            'ori_range': ((16, 31), (3, 3), (16, 31), (16, 31)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_general', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)

def test_conv2d_backprop_input_fuzz_build_upper_limit():
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 2, -1, -1),
            'range': ((32, 2**31 - 1), (1, 1), (16, 31), (1024, 4096), (16, 16)),
            'ori_range': ((32, 2**31 - 1), (2, 2), (16, 31), (1024, 4096)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 2, -1, -1),
            'range': ((32, 2**31 - 1), (1, 1), (32, 63), (1024, 4096), (16, 16)),
            'ori_range': ((32, 2**31 - 1), (2, 2), (32, 63), (1024, 4096)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'test_conv2d_backprop_input_fuzz_build_upper_limit', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)


if __name__ == '__main__':
    test_conv2d_backprop_input_fuzz_build_lower_limit()
    test_conv2d_backprop_input_fuzz_build_upper_limit()