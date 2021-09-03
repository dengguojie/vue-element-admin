#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

from impl.dynamic.conv2d import conv2d_generalization

def test_conv2d_fuzzbuild_generalization():
    input_list = [
        {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 0, 'conv2d_fuzz_build_generalization']
    conv2d_generalization(*input_list)

def test_conv2d_fuzzbuild_generalization_01():
    input_list = [
        {
            'shape': (16, 1, -1, -1, 16),
            'ori_shape': (16, 3, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            "ori_range": [[16,16], [3,3], [16,31], [16,31]]
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (-1, -1, -1, -1), (1, 1, 1, 1), 1, 'NCHW', 0, 'conv2d_fuzz_build_generalization']
    conv2d_generalization(*input_list)

if __name__ == "__main__":
    test_conv2d_fuzzbuild_generalization()
    test_conv2d_fuzzbuild_generalization_01()
