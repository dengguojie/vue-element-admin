#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

avg_pool
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import *
from topi import generic
from topi.cce import util


def get_fusion_params(input_data, output_data, is_fused_compute=True):
    """
    :param input_data: tensor of input_data
    :param output_data: dict of output_data
    :return: dict fusion_params
    """
    # l1 fusion params assign
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in input_data.op.attrs else -1
    in_l1_flag = input_data.op.attrs["addr_type"].value == 1 \
        if "addr_type" in input_data.op.attrs else False
    in_valid_shape = input_data.op.attrs["valid_shape"] \
        if "valid_shape" in input_data.op.attrs else []
    in_slice_offset = input_data.op.attrs["slice_offset"] \
        if "slice_offset" in input_data.op.attrs else []
    in_select_read_flag = bool(in_valid_shape)
    in_split_index = input_data.op.attrs["split_index"].value \
        if "split_index" in input_data.op.attrs else 0
    out_l1_flag = output_data.get("addr_type") == 1
    fusion_params = {"is_fused_compute": is_fused_compute,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag,
                     "out_l1_flag": out_l1_flag,
                     "in_select_read_flag": in_select_read_flag,
                     "in_split_index": in_split_index,
                     "in_slice_offset": in_slice_offset}

    return fusion_params


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,too-many-statements
def check_window_rule(ksize, strides, data_format):
    """
    check ksize and strides of window in pooling
    """
    if data_format in ("NHWC",):
        if len(ksize) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = 'ksize'
            errorInfo['min_value'] = '4'
            errorInfo['max_value'] = '4'
            errorInfo['real_value'] = len(ksize)
            raise RuntimeError(errorInfo,
                               "In op[%s], the num of dimensions of input[%s] should "
                               "be in the range of [%s, %s], but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['min_value'], errorInfo['max_value'],
                                errorInfo['real_value']))

        elif ksize[0] != 1 or ksize[3] != 1:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_000
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = ",".join(("ksize[1]", "ksize[3]"))
            errorInfo['expected_value'] = '1'
            errorInfo['real_value'] = ",".join((ksize[1], ksize[3]))
            raise RuntimeError("In op[%s], the parameter[%s] should be [%s], "
                               "but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['expected_value'], errorInfo['real_value']))
        if len(strides) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = 'strides'
            errorInfo['min_value'] = '4'
            errorInfo['max_value'] = '4'
            errorInfo['real_value'] = len(strides)
            raise RuntimeError(errorInfo,
                               "In op[%s], the num of dimensions of input[%s] should"
                               " be in the range of [%s, %s], but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['min_value'], errorInfo['max_value'],
                                errorInfo['real_value']))
        elif strides[0] != 1 or strides[3] != 1:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_000
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = ",".join(("strides[1]", "strodes[3]"))
            errorInfo['expected_value'] = '1'
            errorInfo['real_value'] = ",".join((strides[1], strides[3]))
            raise RuntimeError("In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['expected_value'], errorInfo['real_value']))
    elif data_format in ("NC1HWC0", "NCHW"):
        if len(ksize) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = 'ksize'
            errorInfo['min_value'] = '4'
            errorInfo['max_value'] = '4'
            errorInfo['real_value'] = len(ksize)
            raise RuntimeError(errorInfo,
                               "In op[%s], the num of dimensions of input[%s] should"
                               " be in the range of [%s, %s], but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['min_value'], errorInfo['max_value'],
                                errorInfo['real_value']))
        elif ksize[0] != 1 or ksize[1] != 1:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_000
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = ",".join(("ksize[0]", "ksize[1]"))
            errorInfo['expected_value'] = '1'
            errorInfo['real_value'] = ",".join((ksize[0], ksize[1]))
            raise RuntimeError("In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['expected_value'], errorInfo['real_value']))
        if len(strides) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = 'strides'
            errorInfo['min_value'] = '4'
            errorInfo['max_value'] = '4'
            errorInfo['real_value'] = len(strides)
            raise RuntimeError(errorInfo,
                               "In op[%s], the num of dimensions of input[%s] should"
                               " be in the range of [%s, %s], but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['min_value'], errorInfo['max_value'],
                                errorInfo['real_value']))
        elif strides[0] != 1 or strides[1] != 1:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_000
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = ",".join(("strides[0]", "strodes[1]"))
            errorInfo['expected_value'] = '1'
            errorInfo['real_value'] = ",".join((strides[1], strides[1]))
            raise RuntimeError("In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['expected_value'], errorInfo['real_value']))
    else:
        errorInfo = {}
        errorInfo['errCode'] = OP_ERROR_CODE_015
        errorInfo['op_name'] = 'avg_pool'
        errorInfo['param_name'] = 'x'
        errorInfo['excepted_format_list'] = ",".join(("NC1HWC0", "NCHW", "NHWC"))
        errorInfo['format'] = data_format
        raise RuntimeError(errorInfo, "In op[%s], the format[%s] of input should"
                                      " be one of [%s], but actually is [%s]."
                           % (errorInfo['op_name'], errorInfo['param_name'],
                              errorInfo['excepted_format_list'], errorInfo['format']))


def get_corrected_pad(input_pad):
    """
    algorithm:
    get corrected pad value

    Parameters
    ----------
    input_pad: the value of pad
    Returns
    -------
    output_pad: the value of pad
    """
    if input_pad < 0:
        output_pad = 0
    else:
        output_pad = input_pad
    return output_pad


def avg_pool_check_rule(input_shape, input_dtype, output_dtype,
                        input_format, ksize, strides,
                        data_format, kernel_name):
    """
    :param input_shape: shape of input_data
    :param input_dtype: dtype of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of avgpooling
    :param strides: the stride of avgpooling window
    :param data_format: NHWC default
    :param kernel_name: cce kernel name
    :return: None

    """
    # check input and output
    check_shape(input_shape, param_name="x")
    check_dtype(input_dtype, ["float16"], param_name="x")
    check_dtype(output_dtype, ["float16"], param_name="y")
    # check ksize and strides of window
    check_window_rule(ksize, strides, data_format)


# pylint: disable=unnecessary-lambda,redefined-builtin,too-many-locals
@fusion_manager.register("avg_pool")
def avg_pool_compute(x, filter, y, ksize, strides, padding="VALID",
                     data_format="NHWC", kernel_name="avg_pool"):
    """
    algorithm: avg_pool
    calculating the average pooling

    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16
    filter : dict, shape and dtype of input_data, only support float16
    y : dict, shape and dtype of output_data, only support float16
    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W
    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W
    padding : str, the mode of padding, support padding and not padding
    data_format : str, default = "NHWC"
    kernel_name : kernel name, default value is "avg_pool"

    Returns
    -------
    None
    """
    out_dtype = y.get("dtype")
    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

    shape_x = x.shape
    input_h = shape_x[2]
    input_w = shape_x[3]
    dilations = (1, 1)
    bias = None
    dsl_flag = False
    offset_x = 0

    if padding == "SAME":
        output_h = (input_h + stride[0] - 1) // stride[0]
        output_w = (input_w + stride[1] - 1) // stride[1]
        pad_row = (output_h - 1) * stride[0] + \
                  ((window[0] - 1) * dilations[0] + 1) - input_h
        pad_col = (output_w - 1) * stride[1] + \
                  ((window[1] - 1) * dilations[1] + 1) - input_w
        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left
        pad_top = get_corrected_pad(int(pad_top))
        pad_bottom = get_corrected_pad(int(pad_bottom))
        pad_left = get_corrected_pad(int(pad_left))
        pad_right = get_corrected_pad(int(pad_right))
        pad = (pad_top, pad_bottom, pad_left, pad_right)
    else:
        pad = (0, 0, 0, 0)
    res = te.lang.cce.te_compute.depthwise_conv2d_compute(
        x, filter, out_dtype.lower(), stride, pad, dilations, {
            "bias_tensor": bias, "dsl_flag": dsl_flag, "offset_x": offset_x},
            kernel_name)

    return res


def avg_pool_compute1(x, y, ksize, strides,
                     padding="VALID", data_format="NHWC",
                     is_fused_compute=True,
                     kernel_name="avg_pool_cce"):
    """
    describe compute
    return: tensor
    """
    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

    window = list(window)
    stride = list(stride)

    # l1 fusion and l2 fusion
    l1_fusion_type = x.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in x.op.attrs else -1
    fusion_params = get_fusion_params(x, y, is_fused_compute)
    in_select_read_flag = fusion_params.get("in_select_read_flag")
    in_valid_shape = fusion_params.get("in_valid_shape")
    in_slice_offset = fusion_params.get("in_slice_offset")

    if in_select_read_flag:
        select_tensor_in = tvm.compute(in_valid_shape,
                                       lambda n, c1, h, w, c0:
                                       x(n, c1, h + in_slice_offset[2], w, c0),
                                       name="tensor_read_select",
                                       attrs=x.op.attrs)
        res = te.lang.cce.pooling2d(select_tensor_in, window, stride, "AVG",
                                    padding, fusion_params=fusion_params)
    elif l1_fusion_type == 1:
        x.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(x.shape,
                                         lambda n, c1, h, w, c0:
                                         x(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=x.op.attrs)
        res = te.lang.cce.pooling2d(l1_width_fusion_in, window, stride,
                                    "AVG", padding,
                                    fusion_params=fusion_params)
    else:
        res = te.lang.cce.pooling2d(x, window, stride, "AVG", padding,
                                    fusion_params=fusion_params)

    return res


@check_op_params(REQUIRED_INPUT, OPTION_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_LIST_INT,
                 REQUIRED_ATTR_STR, OPTION_ATTR_STR, KERNEL_NAME)
def avg_pool(x, filter, y, ksize, strides,
             padding="VALID", data_format="NHWC",
             kernel_name="avg_pool_cce"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    None
    """
    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()

    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    input_format = x.get("format")

    # check others parameter
    avg_pool_check_rule(input_shape, input_dtype, output_dtype,
                       input_format, ksize, strides,
                        data_format, kernel_name)

    # set tensor attrs, during L1 fusion these attrs will assign by te_fusion
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    split_index = x.get("split_index", 0)
    l1_fusion_type = x.get("L1_fusion_type", -1)
    attr = {"addr_type": addr_type,
            "valid_shape": valid_shape,
            "slice_offset": slice_offset,
            "split_index": split_index,
            "L1_fusion_type": l1_fusion_type}
    is_l1fusion = l1_fusion_type in (0, 1)

    if data_format in ("NHWC",):
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        hw = ksize_h * ksize_w
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        hw = ksize_h * ksize_w


    # compute
    # create tensor_in
    tensor_in = tvm.placeholder(input_shape, name="tensor_in",
                                dtype=input_dtype, attrs=attr)
    if filter is not None:
        filter_shape = filter.get("shape")
        filter_dtype = filter.get("dtype").lower()
        filter_c1 = filter_shape[0] / hw
        filter_shape_5d = filter_c1, ksize_h, ksize_w, filter_shape[2],\
                          filter_shape[3]
        filter_in = tvm.placeholder(filter_shape_5d, name="filter_in",
                                    dtype=filter_dtype, attrs=attr)

        res = avg_pool_compute(tensor_in, filter_in, y, ksize, strides,
                               padding, data_format, kernel_name)
        tensor_list = [tensor_in, filter_in, res]
    else:
        res = avg_pool_compute1(tensor_in, y, ksize, strides, padding,
                                data_format, False, kernel_name)

        tensor_list = [tensor_in, res]
    # schedule
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    # build
    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": tensor_list,
              "l1_fusion_option": is_l1fusion}

    te.lang.cce.cce_build_code(sch, config)
