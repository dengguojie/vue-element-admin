#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropFilter fuzzy compile
from impl.dynamic.conv3d_backprop_filter import conv3d_backprop_filter_generalization
from impl.dynamic.conv3d_backprop_filter import UNSUPPORTED_FUZZ_RES


def test_conv3d_backprop_filter_fuzz_compile_head_node_1():
    input_list = [
        {'ori_shape': (1, 16, 28, 30, 32), 'ori_format': 'NDHWC', 'dtype': 'float16',
        'range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32', },
        {'ori_shape': (1, 14, 26, 28, 16), 'ori_format': 'NDHWC', 'dtype': 'float16'},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_head_node_1',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [[
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]],
        'ori_range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 31], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        {'strides': (1, 1, 1, 1, 1)},
        {'pads': (0, 0, 0, 0, 0, 0)},
        {'dilations': (1, 1, 1, 1, 1)},
        {'groups': 1},
        {'data_format': 'NDHWC'}, {'kernel_name': 'test_conv3d_backprop_filter_fuzz_compile_head_node_1'}]]


def test_conv3d_backprop_filter_fuzz_compile_head_node_2():
    # N dim exceeds limit
    input_list = [
        {'ori_shape': (2147483648, 16, 28, 30, 32), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32', },
        {'ori_shape': (2147483648, 14, 26, 28, 16), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32' },
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_head_node_2',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_conv3d_backprop_filter_fuzz_compile_head_node_3():
    # D dim exceeds limit
    input_list = [
        {'ori_shape': (1, 4900, 28, 30, 32), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32', },
        {'ori_shape': (1, 4898, 26, 28, 16), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32' },
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_head_node_3',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_conv3d_backprop_filter_fuzz_compile_head_node_4():
    # HW dim exceeds limit
    input_list = [
        {'ori_shape': (1, 16, 4800, 30, 32), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32', },
        {'ori_shape': (1, 14, 4798, 28, 16), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32' },
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_head_node_4',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_conv3d_backprop_filter_fuzz_compile_head_node_5():
    # [-2]
    input_list = [
        {'ori_shape': (-2,), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32', },
        {'ori_shape': (1, 14, 26, 28, 16), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32' },
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_head_node_5',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_conv3d_backprop_filter_fuzz_compile_head_node_6():
    # fmap exceeds l1 size
    input_list = [
        {'ori_shape': (1, 16, 30, 2000, 32), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32', },
        {'ori_shape': (1, 14, 13, 1983, 16), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (3, 18, 18, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32' },
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_head_node_6',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_conv3d_backprop_filter_fuzz_compile_head_node_7():
    # fmap exceeds l1 size, modify w range
    input_list = [
        {'ori_shape': (1, 16, 30, 1617, 32), 'ori_format': 'NDHWC', 'dtype': 'float16',
        'range': [[1, 1], [16, 31], [18, 31], [1024, 1723], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32', },
        {'ori_shape': (1, 14, 13, 1600, 16), 'ori_format': 'NDHWC', 'dtype': 'float16' },
        {'ori_shape': (3, 18, 18, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32' },
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_head_node_7',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [[
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [16, 31], [18, 31], [1024, 1723], [32, 32]],
        'range': [[1, 1], [16, 31], [18, 31], [1024, 1723], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [4, 15], [1024, 4096], (16, 16)]},
        {'ori_shape': (3, 18, 18, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        {'strides': (1, 1, 1, 1, 1)}, {'pads': (0, 0, 0, 0, 0, 0)},
        {'dilations': (1, 1, 1, 1, 1)}, {'groups': 1}, {'data_format': 'NDHWC'},
        {'kernel_name': 'test_conv3d_backprop_filter_fuzz_compile_head_node_7'}]]


def test_conv3d_backprop_filter_fuzz_compile_body_node_1():
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]],
        'range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 31], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_1',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [[
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]],
        'range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 31], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        {'strides': (1, 1, 1, 1, 1)}, {'pads': (0, 0, 0, 0, 0, 0)},
        {'dilations': (1, 1, 1, 1, 1)}, {'groups': 1}, {'data_format': 'NDHWC'},
        {'kernel_name': 'test_conv3d_backprop_filter_fuzz_compile_body_node_1'}]]


def test_conv3d_backprop_filter_fuzz_compile_body_node_2():
    # Invalid range length
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [16, 31], [16, 31], [16, 31]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 31], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_2',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_conv3d_backprop_filter_fuzz_compile_body_node_3():
    # fmap N dim upper bound exceeds limit
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 2147483648], [16, 31], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 31], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_3',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [{'result': 'UNSUPPORTED', 'reason': {'param_index': [0], 'type': ['upper_limit']}}]


def test_conv3d_backprop_filter_fuzz_compile_body_node_4():
    # dedy H dim upper bound exceeds limit
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 4500], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_4',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [{'result': 'UNSUPPORTED', 'reason': {'param_index': [2], 'type': ['upper_limit']}}]


def test_conv3d_backprop_filter_fuzz_compile_body_node_5():
    # fmap D dim lower bound exceeds limit
    # dedy H dim upper bound exceeds limit
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4600, 31], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 4500], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_5',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [{'result': 'UNSUPPORTED', 'reason': {'param_index': [0, 2], 'type': ['lower_limit', 'upper_limit']}}]


def test_conv3d_backprop_filter_fuzz_compile_body_node_6():
    # fmap D dim upper bound exceeds limit
    # dedy H dim upper bound exceeds limit
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [30, 4400], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 4500], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_6',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [{'result': 'UNSUPPORTED', 'reason': {'param_index': [0, 2], 'type': ['upper_limit', 'upper_limit']}}]


def test_conv3d_backprop_filter_fuzz_compile_body_node_7():
    # fmap w dim lower bound exceeds limit
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [16, 31], [16, 31], [4400, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 31], [16, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_7',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [{'result': 'UNSUPPORTED', 'reason': {'param_index': [0], 'type': ['lower_limit']}}]


def test_conv3d_backprop_filter_fuzz_compile_body_node_8():
    # dedy w dim lower bound exceeds limit
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [16, 31], [16, 31], [16, 31], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [16, 31], [4311, 31], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_8',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [{'result': 'UNSUPPORTED', 'reason': {'param_index': [2], 'type': ['lower_limit']}}]


def test_conv3d_backprop_filter_fuzz_compile_body_node_9():
    # upper bound is -1
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, -1], [16, -1], [16, -1], [16, -1], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, -1], [4, -1], [16, -1], [16, -1], (16, 16)]},
        {'ori_shape': (3, 3, 3, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_9',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [{'result': 'UNSUPPORTED', 'reason': {'param_index': [0, 2], 'type': ['upper_limit', 'upper_limit']}}]


def test_conv3d_backprop_filter_fuzz_compile_body_node_10():
    # fmap exceeds l1 size
    input_list = [
        {'ori_shape': [-1, -1, -1, -1, 32], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [16, 31], [18, 31], [1024, 1724], [32, 32]]},
        {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        {'ori_shape': [-1, -1, -1, -1, 16], 'ori_format': 'NDHWC', 'dtype': 'float16',
        'ori_range': [[1, 1], [4, 15], [4, 15], [1024, 4096], (16, 16)]},
        {'ori_shape': (3, 18, 18, 32, 16), 'ori_format': 'DHWCN', 'dtype': 'float32'},
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1), 1, 'NDHWC', 'test_conv3d_backprop_filter_fuzz_compile_body_node_10',
        {"mode": "keep_rank"}]

    res = conv3d_backprop_filter_generalization(*input_list)
    assert res == [{'result': 'UNSUPPORTED', 'reason': {'param_index': [0], 'type': ['upper_limit']}}]


if __name__ == '__main__':
    test_conv3d_backprop_filter_fuzz_compile_head_node_1()
    test_conv3d_backprop_filter_fuzz_compile_head_node_2()
    test_conv3d_backprop_filter_fuzz_compile_head_node_3()
    test_conv3d_backprop_filter_fuzz_compile_head_node_4()
    test_conv3d_backprop_filter_fuzz_compile_head_node_5()
    test_conv3d_backprop_filter_fuzz_compile_head_node_6()
    test_conv3d_backprop_filter_fuzz_compile_head_node_7()
    test_conv3d_backprop_filter_fuzz_compile_body_node_1()
    test_conv3d_backprop_filter_fuzz_compile_body_node_2()
    test_conv3d_backprop_filter_fuzz_compile_body_node_3()
    test_conv3d_backprop_filter_fuzz_compile_body_node_4()
    test_conv3d_backprop_filter_fuzz_compile_body_node_5()
    test_conv3d_backprop_filter_fuzz_compile_body_node_6()
    test_conv3d_backprop_filter_fuzz_compile_body_node_7()
    test_conv3d_backprop_filter_fuzz_compile_body_node_8()
    test_conv3d_backprop_filter_fuzz_compile_body_node_9()
    test_conv3d_backprop_filter_fuzz_compile_body_node_10()