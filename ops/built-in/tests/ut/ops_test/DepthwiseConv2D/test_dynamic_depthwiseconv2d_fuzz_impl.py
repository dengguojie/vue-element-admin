#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.dynamic.depthwise_conv2d import depthwise_conv2d_generalization

ut_case = OpUT("DepthwiseConv2D", "impl.dynamic.depthwise_depthwise", "depthwise_depthwise")


def test_depthwise_fuzz_build_graph_mode_check_range(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (32, 63), (64, 127), (64, 127)),
            'ori_range': ((1, 1), (32, 63), (64, 127), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 96, 1),
            'ori_shape': (3, 3, 96, 1),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 96),
            'ori_shape': (1, -1, -1, 96),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_check_range']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_check_range testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_check_range)


def test_depthwise_fuzz_build_graph_mode_check_range_nchw(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'float32',
            'range': ((1, 1), (32, 63), (64, 127), (64, 127)),
            'ori_range': ((1, 1), (32, 63), (64, 127), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 96, 1),
            'ori_shape': (3, 3, 96, 1),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 96),
            'ori_shape': (1, -1, -1, 96),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_check_range_nchw']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_check_range_nchw testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_check_range_nchw)


def test_depthwise_fuzz_build_graph_mode_same_mode(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (32, 63), (64, 127), (64, 127)),
            'ori_range': ((1, 1), (32, 63), (64, 127), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 96, 1),
            'ori_shape': (3, 3, 96, 1),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 96),
            'ori_shape': (1, -1, -1, 96),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_same_mode']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_same_mode testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_same_mode)


def test_depthwise_fuzz_build_graph_mode_whcn_exception(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'WHCN',
            'format': 'WHCN',
            'dtype': 'float32',
            'range': ((1, 1), (32, 63), (64, 127), (64, 127)),
            'ori_range': ((1, 1), (32, 63), (64, 127), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 96, 1),
            'ori_shape': (3, 3, 96, 1),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 96),
            'ori_shape': (1, -1, -1, 96),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_whcn_exception']
    try:
        depthwise_conv2d_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding depthwise test_depthwise_fuzz_build_graph_mode_whcn_exception testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_whcn_exception)


def test_depthwise_fuzz_build_graph_mode_below_upper_limit(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (64, 128), (64, 504), (64, 127)),
            'ori_range': ((1, 1), (64, 128), (64, 504), (64, 127))
        }, {
            # filter
            'shape': (64, 64, 3, 256),
            'ori_shape': (64, 64, 3, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_upper_limit']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_below_upper_limit testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_below_upper_limit)


def test_depthwise_fuzz_build_graph_mode_upper_limit(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (64, 128), (64, 505), (64, 127)),
            'ori_range': ((1, 1), (64, 128), (64, 505), (64, 127))
        }, {
            # filter
            'shape': (64, 64, 3, 256),
            'ori_shape': (64, 64, 3, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_upper_limit']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_upper_limit testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_upper_limit)


def test_depthwise_fuzz_build_graph_mode_below_upper_limit_same_mode(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (64, 96), (16, 512), (64, 127)),
            'ori_range': ((1, 1), (64, 96), (16, 512), (64, 127))
        }, {
            # filter
            'shape': (63, 3, 3, 256),
            'ori_shape': (63, 3, 3, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_upper_limit']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_below_upper_limit_same_mode testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_below_upper_limit_same_mode)


def test_depthwise_fuzz_build_graph_mode_upper_limit_same_mode(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range':  ((1, 1), (16, 96), (16, 513), (64, 127)),
            'ori_range': ((1, 1), (16, 96), (16, 513), (64, 127))
        }, {
            # filter
            'shape': (63, 3, 3, 256),
            'ori_shape': (63, 3, 3, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_upper_limit']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_upper_limit_same_mode testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_upper_limit_same_mode)


def test_depthwise_fuzz_build_graph_mode_up_low_limit_h(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (16, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, 31), (16, 31), (64, 127))
        }, {
            # filter
            'shape': (18, 3, 3, 256),
            'ori_shape': (18, 3, 3, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_low_limit']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_up_low_limit_h testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_up_low_limit_h)


def test_depthwise_fuzz_build_graph_mode_low_limit_h(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (16, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, 31), (16, 31), (64, 127))
        }, {
            # filter
            'shape': (19, 3, 3, 256),
            'ori_shape': (19, 3, 3, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_low_limit']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_low_limit_h testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_low_limit_h)


def test_depthwise_fuzz_build_graph_mode_up_low_limit_w(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (16, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, 31), (16, 31), (64, 127))
        }, {
            # filter
            'shape': (3, 18, 3, 256),
            'ori_shape': (3, 18, 3, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_low_limit_w']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_up_low_limit_w testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_up_low_limit_w)


def test_depthwise_fuzz_build_graph_mode_low_limit_w(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (16, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, 31), (16, 31), (64, 127))
        }, {
            # filter
            'shape': (3, 19, 3, 256),
            'ori_shape': (3, 19, 3, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_low_limit_w']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_low_limit_w testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_low_limit_w)


def test_depthwise_fuzz_build_graph_mode_range_exception_01(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (16, -1), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, -1), (16, 31), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 64, 256),
            'ori_shape': (3, 3, 64, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_range_exception']
    try:
        depthwise_conv2d_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding depthwise test_depthwise_fuzz_build_graph_mode_range_exception_01 testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_range_exception_01)


def test_depthwise_fuzz_build_graph_mode_range_exception_02(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (-1, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (-1, 31), (16, 31), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 64, 256),
            'ori_shape': (3, 3, 64, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), 'NHWC', 0, 'test_depthwise_fuzz_build_graph_mode_range_exception']
    try:
        depthwise_conv2d_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding depthwise test_depthwise_fuzz_build_graph_mode_range_exception_02 testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_range_exception_02)


def test_depthwise_fuzz_build_graph_mode_config_exception(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (16, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, 31), (16, 31), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 64, 256),
            'ori_shape': (3, 3, 64, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, groups, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), 'NHWC', 0,
        'test_depthwise_fuzz_build_graph_mode_config_exception', {"mode": "keep_rank_00"}]
    try:
        depthwise_conv2d_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding depthwise test_depthwise_fuzz_build_graph_mode_config_exception testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_config_exception)


def test_depthwise_fuzz_build_graph_mode_unknown_rank_exception(test_arg):
    input_list = [
        {
            # x
            'shape': [-2],
            'ori_shape': [-2],
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (16, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, 31), (16, 31), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 64, 256),
            'ori_shape': (3, 3, 64, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, data_format, offset_x, kernel_name
        (1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), 'NHWC', 0,
        'test_depthwise_fuzz_build_graph_mode_unknown_rank_exception']
    try:
        depthwise_conv2d_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding depthwise test_depthwise_fuzz_build_graph_mode_unknown_rank_exception testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_unknown_rank_exception)


def test_depthwise_fuzz_build_graph_mode_none_range(test_arg):
    input_list = [
        {
            # x
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NHWC',
            'dtype': 'float32',
            'range': ((1, 1), (16, None), (16, None), (64, 127)),
            'ori_range': ((1, 1), (16, None), (16, None), (64, 127))
        }, {
            # filter
            'shape': (3, 3, 64, 256),
            'ori_shape': (3, 3, 64, 256),
            'format': 'HWCN',
            'ori_format': 'HWCN',
            'dtype': 'float32'
        }, None, None, {
            # y
            'shape': (1, -1, -1, 256),
            'ori_shape': (1, -1, -1, 256),
            'ori_format': 'HWCN',
            'format': 'NHWC',
            'dtype': 'float32'
        },
        # strides, dilations, pads, groups, data_format, offset_x, kernel_name
        (1, 1, 100, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NHWC', 0,
        'test_depthwise_fuzz_build_graph_mode_none_range']
    depthwise_conv2d_generalization(*input_list)
print("adding depthwise test_depthwise_fuzz_build_graph_mode_none_range testcase")
ut_case.add_cust_test_func(test_func=test_depthwise_fuzz_build_graph_mode_none_range)


if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend310"])
    exit(0)
