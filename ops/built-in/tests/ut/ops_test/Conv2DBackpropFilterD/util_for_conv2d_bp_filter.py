#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the Conv2DTransposeD util test
"""
from math import ceil as math_ceil


def get_block(val, dtype):
    """ function : get the miniest block size """
    assert val in ("m", "n", "k")
    if val in ("m", "n"):
        return 16
    return 32 if dtype in ["int8", "uint8"] else 16


def ceil(val, base):
    """ function : ceil """
    return math_ceil(val / base)


def align(val, base):
    """ function : align """
    return ceil(val, base) * base


def trans_to_nchw(shape, data_format):
    """ trans data format from NHWC to NCHW """
    if data_format == "NHWC":
        return [
            shape[data_format.index("N")],
            shape[data_format.index("C")],
            shape[data_format.index("H")],
            shape[data_format.index("W")],
        ]
    return shape


def shape_4d_to_5hd(shape, dtype, data_format):
    """ trans data from 4d to NC1HWC0 """
    if len(shape) != 4:
        return shape
    shape_nc1hwc0 = list(trans_to_nchw(shape, data_format))
    block_k = get_block("k", dtype)
    shape_nc1hwc0[1] = align(shape_nc1hwc0[1], block_k)
    return (
        shape_nc1hwc0[0],
        shape_nc1hwc0[1] // block_k,
        shape_nc1hwc0[2],
        shape_nc1hwc0[3],
        block_k,
    )


def shape_4d_to_fz(shape, dtype, data_format):
    """ trans data from 4d to fz """
    shape_nchw = trans_to_nchw(shape, data_format)
    if len(shape_nchw) != 4:
        return shape
    if dtype == "int8":
        shape_nchw = exchange_filter_nc_axis(shape, data_format)

    c_out, c_in, height, weight = shape_nchw
    block_k = get_block("k", dtype)
    block_n = get_block("n", dtype)
    c_in1 = ceil(c_in, block_k)
    c_in0 = block_k
    c_out1 = ceil(c_out, block_n)
    c_out0 = block_n
    return c_in1 * height * weight, c_out1, c_out0, c_in0


def get_ori_shape(shape, dtype, data_format):
    """ get ori shape """
    if dtype == "int8":
        return exchange_filter_nc_axis(shape, data_format)
    return shape


def exchange_filter_nc_axis(ori_shape_filters, ori_format_filters):
    """ exchange nc axis when data type int8 """
    if ori_format_filters == "NCHW":
        return (
            ori_shape_filters[1],
            ori_shape_filters[0],
            ori_shape_filters[2],
            ori_shape_filters[3],
        )
    elif ori_format_filters == "NHWC":
        return (
            ori_shape_filters[3],
            ori_shape_filters[1],
            ori_shape_filters[2],
            ori_shape_filters[0],
        )
    elif ori_format_filters == "HWCN":
        return (
            ori_shape_filters[0],
            ori_shape_filters[1],
            ori_shape_filters[3],
            ori_shape_filters[2],
        )
    else:
        return (
            ori_shape_filters[0],
            ori_shape_filters[1],
            ori_shape_filters[2],
            ori_shape_filters[3],
        )


def gen_padding_size(
    shape_x,
    shape_out_backprop,
    filter_sizes,
    padding,
    strides,
    dilations,
    data_format="NCHW",
    filter_format="NCHW",
):
    """ gen padding size """
    if isinstance(padding, (list, tuple)):
        return padding
    else:
        # TO DO compute by format
        _, fmap_channel, fmap_h, fmap_w = shape_x
        _, _, dedy_h, dedy_w = shape_out_backprop
        _, _, filter_h, filter_w = filter_sizes
        _, _, stride_h, stride_w = strides
        _, dilation_c, dilation_h, dilation_w = dilations

        filter_h_dilation = (filter_h - 1) * dilation_h + 1
        filter_w_dilation = (filter_w - 1) * dilation_w + 1
        if padding == "SAME":
            pad_w = align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
            pad_w = max(pad_w, 0)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_h = align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
            pad_h = max(pad_h, 0)
            pad_up = pad_h // 2
            pad_down = pad_h - pad_up
            padding = [pad_up, pad_down, pad_left, pad_right]
        elif padding == "VALID":
            padding = [0, 0, 0, 0]
        return padding
