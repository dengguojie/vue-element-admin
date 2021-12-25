#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

from impl.dynamic.avg_pool_v2 import avg_pool_v2_generalization

# supported range
# NCHW, valid
def test_avg_pool_v2_generalization_graph_mode_01():
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
            # weight
            'ori_shape': (16, 1, 5, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (5, 5), (5, 5)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 1, 5, 5), (1, 1, 2, 2), "VALID", None, 'NCHW', None, None, None, 0,
        'test_avg_pool_v2_generalization_graph_mode_01']
    avg_pool_v2_generalization(*input_list)

# supported range
# NCHW, same
def test_avg_pool_v2_generalization_graph_mode_02():
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
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 1, 3, 3), (1, 1, 1, 1), "SAME", None, 'NCHW', None, None, None, 0,
        'test_avg_pool_v2_generalization_graph_mode_02']
    avg_pool_v2_generalization(*input_list)

# supported range
# NCHW, calculated
def test_avg_pool_v2_generalization_graph_mode_03():
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
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 2, 2), 'NCHW', None, None, None, 0,
        'test_avg_pool_v2_generalization_graph_mode_03']
    avg_pool_v2_generalization(*input_list)

# supported range
# NHWC, valid
def test_avg_pool_v2_generalization_graph_mode_04():
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
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 5, 5, 1), (1, 2, 2, 1), "VALID", None, 'NHWC', None, None, None, 0,
        'test_avg_pool_v2_generalization_graph_mode_04']
    avg_pool_v2_generalization(*input_list)

# supported range
# NHWC, same
def test_avg_pool_v2_generalization_graph_mode_05():
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
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 3, 3, 1), (1, 1, 1, 1), "SAME", None, 'NHWC', None, None, None, 0,
        'test_avg_pool_v2_generalization_graph_mode_05']
    avg_pool_v2_generalization(*input_list)

# supported range
# NHWC, calculated
def test_avg_pool_v2_generalization_graph_mode_06():
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
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 3, 3, 1), (1, 1, 1, 1), "CALCULATED", (1, 1, 2, 2), 'NHWC', None, None, None, 0,
        'test_avg_pool_v2_generalization_graph_mode_06']
    avg_pool_v2_generalization(*input_list)

if __name__ == "__main__":
    test_avg_pool_v2_generalization_graph_mode_01()
    test_avg_pool_v2_generalization_graph_mode_02()
    test_avg_pool_v2_generalization_graph_mode_03()
    test_avg_pool_v2_generalization_graph_mode_04()
    test_avg_pool_v2_generalization_graph_mode_05()
    test_avg_pool_v2_generalization_graph_mode_06()