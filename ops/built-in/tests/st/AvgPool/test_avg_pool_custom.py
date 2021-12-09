#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

from impl.dynamic.avg_pool import avg_pool_generalization

def test_avg_pool_fuzzbuild_generalization_01():
    input_list = [
        {
            'shape': (16, 3, 16, 16, 16),
            'ori_shape': (16, 33, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (33, 1, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 3, 5), (1, 1, 1, 1), "VALID", 'NCHW', 0, 'avg_pool_fuzz_build_generalization']
    avg_pool_generalization(*input_list)

def test_avg_pool_fuzzbuild_generalization_02():
    input_list = [
        {
            'shape': (1, 4, 1080, 1080, 16),
            'ori_shape': (1, 64, 1080, 1080),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 2), (1, 1), (1024, 4096), (1024, 4096), (16, 16)),
            'ori_range': ((1, 2), (3, 3), (1024, 4096), (1024, 4096))
            }, {
                'ori_shape': (64, 64, 7, 7),
                'ori_format': 'NCHW',
                'format': 'FRACTAL_Z',
                'dtype': 'float16'
            }, None, {
            'shape': (-1, 4, -1, -1, 16),
            'ori_shape': (-1, 64, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 7, 7), (1, 1, 2, 2), "SAME", 'NCHW', 0, 'test_avg_pool_fuzz_build_correct_range']
    avg_pool_generalization(*input_list)

# supported range
# NCHW, valid
def test_avg_pool_fuzzbuild_generalization_graph_mode_01():
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 5, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 1, 5, 5), (1, 1, 2, 2), "VALID", 'NCHW', 0, 'test_avg_pool_fuzzbuild_generalization_graph_mode_01']
    avg_pool_generalization(*input_list)

# supported range
# NCHW, same
def test_avg_pool_fuzzbuild_generalization_graph_mode_02():
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (128, 255), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (128, 255), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 1, 3, 3), (1, 1, 1, 1), "SAME", 'NCHW', 0, 'test_avg_pool_fuzzbuild_generalization_graph_mode_02']
    avg_pool_generalization(*input_list)

# supported range
# NHWC, valid
def test_avg_pool_fuzzbuild_generalization_graph_mode_03():
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 5, 5, 1), (1, 2, 2, 1), "VALID", 'NHWC', 0, 'test_avg_pool_fuzzbuild_generalization_graph_mode_03']
    avg_pool_generalization(*input_list)

# supported range
# NHWC, same
def test_avg_pool_fuzzbuild_generalization_graph_mode_04():
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (128, 255), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (128, 255), (16, 31), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 3, 3, 1), (1, 1, 1, 1), "SAME", 'NHWC', 0, 'test_avg_pool_fuzzbuild_generalization_graph_mode_04']
    avg_pool_generalization(*input_list)

if __name__ == "__main__":
    test_avg_pool_fuzzbuild_generalization_01()
    test_avg_pool_fuzzbuild_generalization_02()
    test_avg_pool_fuzzbuild_generalization_graph_mode_01()
    test_avg_pool_fuzzbuild_generalization_graph_mode_02()
    test_avg_pool_fuzzbuild_generalization_graph_mode_03()
    test_avg_pool_fuzzbuild_generalization_graph_mode_04()
