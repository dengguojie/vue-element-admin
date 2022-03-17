#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.conv3d import conv3d_generalization


def test_conv3d_fuzz_build_static():
    input_list = [
        {'shape': (2, 8, 8, 8, 320),
         'ori_shape': (2, 8, 8, 8, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (2, 4, 4, 4, 320),
         'ori_shape': (2, 4, 4, 4, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'}, (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_static_mode_general_case', {"mode": "keep_rank"}
    ]
    conv3d_generalization(*input_list)

def test_conv3d_fuzz_build_dynamic_general():
    input_list = [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (4, 10), (320, 320)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (4, 10), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_general_case', {"mode": "keep_rank"}
    ]
    conv3d_generalization(*input_list)

def test_conv3d_fuzz_build_static_unsupport():
    input_list = [
        {'ori_shape': (1, 44, 24, 8, 8),
         'ori_format': 'NCDHW',
         'dtype': 'float16'},
        {'ori_shape': (112, 5, 1, 7, 44),
         'ori_format': 'NDHWC',
         'dtype': 'float16'}, None, None,
        {'ori_shape': (1, 112, 8, 3, 3),
         'ori_format': 'NCDHW',
         'dtype': 'float16'}, (1, 1, 3, 3, 3), (1, 1, 0, 0, 2, 3), (1, 1, 1, 1, 1), 1, 'NCDHW', 0,
        'test_conv3d_generalization_static_mode_w_unsupport_case', {"mode": "keep_rank"}
    ]
    conv3d_generalization(*input_list)

def test_conv3d_fuzz_build_static_modify_range():
    input_list = [
        {'ori_shape': (1, 44, 24, 8, 10),
         'ori_format': 'NCDHW',
         'dtype': 'float16'},
        {'ori_shape': (112, 5, 1, 7, 44),
         'ori_format': 'NDHWC',
         'dtype': 'float16'}, None, None,
        {'ori_shape': (1, 112, 8, 3, 3),
         'ori_format': 'NCDHW',
         'dtype': 'float16'}, (1, 1, 3, 3, 3), (1, 1, 0, 0, 2, 3), (1, 1, 1, 1, 1), 1, 'NCDHW', 0,
        'test_conv3d_generalization_static_mode_w_range_modify_case', {"mode": "keep_rank"}
    ]
    conv3d_generalization(*input_list)

if __name__ == '__main__':
    test_conv3d_fuzz_build_static()
    test_conv3d_fuzz_build_dynamic_general()
    test_conv3d_fuzz_build_static_unsupport()
    test_conv3d_fuzz_build_static_modify_range()