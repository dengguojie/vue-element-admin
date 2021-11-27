#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.conv2d_backprop_filter import conv2d_bp_filter_generalization


def test_conv2d_backprop_filter_fuzz_build_upper_limit():
    input_list = [
        {
            'shape': (-1, -1, -1, 2),
            'ori_shape': (-1, -1, -1, 2),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float16',
            'ori_range': ((2, 3), (128, 191), (256, 2511), (2, 2)),
            'range': ((2, 3), (128, 191), (256, 2511), (2, 2)),
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (-1, -1, -1, 16),
            'ori_shape': (-1, -1, -1, 16),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float16',
            'ori_range': ((2, 3), (128, 191), (256, 2511), (16, 16)),
            'range': ((2, 3), (128, 191), (256, 2511), (16, 16))
        }, {
            'ori_shape': (24, 8, 2, 16),
            'ori_format': 'HWCN',
            'format': 'HWCN',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NHWC', 'conv2d_backprop_filter_fuzz_build_generalization',
        {"mode": "keep_rank"}]
    conv2d_bp_filter_generalization(*input_list)


def test_conv2d_backprop_filter_fuzz_build_range_check_pass():
    input_list = [
        {
            'shape': (-1, -1, -1, 2),
            'ori_shape': (-1, -1, -1, 2),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float16',
            'ori_range': ((2, 3), (128, 191), (256, 511), (2, 2)),
            'range': ((2, 3), (128, 191), (256, 511), (2, 2)),
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (-1, -1, -1, 16),
            'ori_shape': (-1, -1, -1, 16),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float16',
            'ori_range': ((2, 3), (128, 191), (256, 511), (16, 16)),
            'range': ((2, 3), (128, 191), (256, 511), (16, 16))
        }, {
            'ori_shape': (24, 8, 2, 16),
            'ori_format': 'HWCN',
            'format': 'HWCN',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NHWC', 'conv2d_backprop_filter_fuzz_build_generalization',
        {"mode": "keep_rank"}]
    conv2d_bp_filter_generalization(*input_list)


def test_conv2d_backprop_filter_fuzz_build_lower_limit():
    input_list = [
        {
            'shape': (-1, -1, -1, 2),
            'ori_shape': (-1, -1, -1, 2),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float16',
            'ori_range': ((2, 3), (128, 191), (2256, 2511), (2, 2)),
            'range': ((2, 3), (128, 191), (2256, 2511), (2, 2)),
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (-1, -1, -1, 16),
            'ori_shape': (-1, -1, -1, 16),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float16',
            'ori_range': ((2, 3), (128, 191), (2256, 2511), (16, 16)),
            'range': ((2, 3), (128, 191), (2256, 2511), (16, 16))
        }, {
            'ori_shape': (24, 8, 2, 16),
            'ori_format': 'HWCN',
            'format': 'HWCN',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NHWC', 'conv2d_backprop_filter_fuzz_build_generalization',
        {"mode": "keep_rank"}]
    conv2d_bp_filter_generalization(*input_list)


if __name__ == '__main__':
    test_conv2d_backprop_filter_fuzz_build_range_check_pass()
    test_conv2d_backprop_filter_fuzz_build_upper_limit()
    test_conv2d_backprop_filter_fuzz_build_lower_limit()