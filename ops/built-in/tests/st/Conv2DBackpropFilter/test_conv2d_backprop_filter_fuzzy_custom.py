#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.conv2d_backprop_filter import conv2d_bp_filter_generalization
from impl.dynamic.conv2d_backprop_filter import conv2d_backprop_filter_fusion_compute
from impl.dynamic.trans_data import trans_data_fusion_compute
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import tvm

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


def test_conv2d_backprop_filter_binary_mode():
    batch_idx = operation.var("batch")
    fmap_c = operation.var("fmap_c")
    fmap_h = operation.var("fmap_h")
    fmap_w = operation.var("fmap_w")
    dedy_c = operation.var("dedy_c")
    dedy_h = operation.var("dedy_h")
    dedy_w = operation.var("dedy_w")

    fmap_nchw = (batch_idx, fmap_c, fmap_h, fmap_w)
    dedy_nchw = (batch_idx, dedy_c, dedy_h, dedy_w)
    fmap_nc_hw = (batch_idx, fmap_c, fmap_h * fmap_w)
    dedy_nc_hw = (batch_idx, dedy_c, dedy_h * dedy_w)

    fmap = tvm.placeholder(fmap_nc_hw, name="fmap", dtype="float16", attrs={"shape": fmap_nchw})
    dedy = tvm.placeholder(dedy_nc_hw, name="dedy", dtype="float16", attrs={"shape": dedy_nchw})
    filter_tensor = tvm.placeholder([4,], name="filter_tensor", dtype="int32")
    y = {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NCHW"}
    strides = (1, 1, 1, 1)
    pads = (0, 0, 0, 0)
    dilations = (1, 1, 1, 1)

    fmap_nc1hwc0 = trans_data_fusion_compute(fmap, None, "NCHW", "NC1HWC0")
    dedy_nc1hwc0 = trans_data_fusion_compute(dedy, None, "NCHW", "NC1HWC0")

    dedw = conv2d_backprop_filter_fusion_compute(fmap_nc1hwc0, filter_tensor, dedy_nc1hwc0, y, strides, pads, dilations)


if __name__ == '__main__':
    test_conv2d_backprop_filter_fuzz_build_range_check_pass()
    test_conv2d_backprop_filter_fuzz_build_upper_limit()
    test_conv2d_backprop_filter_fuzz_build_lower_limit()
    test_conv2d_backprop_filter_binary_mode()