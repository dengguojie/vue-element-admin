#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from impl.dynamic.conv2d_transpose import conv2d_transpose_generalization
"""
the test_conv2d_transpose_fuzz_build_generalization test
"""

def test_conv2d_transpose_fuzz_build_generalization():
    input_list_same = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (10, 1, 7, 7, 16),
            'ori_shape': (10, 16, 7, 7),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (16, 16, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'shape': (10, 1, 7, 7, 16),
            'ori_shape': (10, 16, 7, 7),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', (0, 0, 0, 0), 0,
        'conv2d_transpose_fuzz_build_generalization_same']
    conv2d_transpose_generalization(*input_list_same)
    input_list_valid = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (10, 1, 5, 5, 16),
            'ori_shape': (10, 16, 5, 5),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (16, 16, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'shape': (10, 1, 7, 7, 16),
            'ori_shape': (10, 16, 7, 7),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', (0, 0, 0, 0), 0,
        'conv2d_transpose_fuzz_build_generalization_valid']
    conv2d_transpose_generalization(*input_list_valid)

if __name__ == "__main__":
    test_conv2d_transpose_fuzz_build_generalization()