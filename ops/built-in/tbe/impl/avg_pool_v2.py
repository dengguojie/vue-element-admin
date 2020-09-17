#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
avg_pool_v2
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
def check_window_rule(ksize, strides, pads, data_format):
    """
    check ksize and strides of window in pooling
    """
    if len(pads) != 4:
        errorInfo = {}
        errorInfo['errCode'] = OP_ERROR_CODE_012
        errorInfo['op_name'] = 'avg_pool_v2'
        errorInfo['param_name'] = 'pads'
        errorInfo['min_value'] = '4'
        errorInfo['max_value'] = '4'
        errorInfo['real_value'] = len(pads)
        raise RuntimeError(errorInfo,
                           "In op[%s], the num of dimensions of input[%s] should"
                           " be in the range of [%s, %s], but actually is [%s]." %
                           (errorInfo['op_name'], errorInfo['param_name'],
                            errorInfo['min_value'], errorInfo['max_value'],
                            errorInfo['real_value']))
    if data_format in ("NHWC",):
        if len(ksize) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool_v2'
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
            errorInfo['op_name'] = 'avg_pool_v2'
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
            errorInfo['op_name'] = 'avg_pool_v2'
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
            errorInfo['op_name'] = 'avg_pool_v2'
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
            errorInfo['op_name'] = 'avg_pool_v2'
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
            errorInfo['op_name'] = 'avg_pool_v2'
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
            errorInfo['op_name'] = 'avg_pool_v2'
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
            errorInfo['op_name'] = 'avg_pool_v2'
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
        errorInfo['op_name'] = 'avg_pool_v2'
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


def avg_pool_v2_check_rule(input_shape, input_dtype, output_dtype,
                           input_format, ksize, strides, pads,
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
    check_window_rule(ksize, strides, pads, data_format)


def avg_pool_v2_compute1(x, y, ksize, strides, padding, pads, data_format,
                         global_pooling, ceil_mode, exclusive, is_fused_compute=True,
                         kernel_name="avg_pool_v2"):
    """
    describe compute
    return: tensor
    """
    input_shape = te.lang.cce.util.shape_to_list(x.shape)
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

    if global_pooling:
        window = [input_shape[2], input_shape[3]]
        padding = "VALID"

    if padding == "CALCULATED":
        if ceil_mode:
            ceil_mode = 0
        else:
            ceil_mode = 1

    if in_select_read_flag:
        select_tensor_in = tvm.compute(in_valid_shape,
                                       lambda n, c1, h, w, c0:
                                       x(n, c1, h + in_slice_offset[2], w, c0),
                                       name="tensor_read_select",
                                       attrs=x.op.attrs)
        if padding == "CALCULATED":
            res = te.lang.cce.pooling2d(select_tensor_in, window, stride, "AVG", "CALCULATED",
                                        pad=pads, data_mode=1, ceil_mode=ceil_mode,
                                        fusion_params=fusion_params)
        else:
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
        if padding == "CALCULATED":
            res = te.lang.cce.pooling2d(l1_width_fusion_in, window, stride, "AVG", "CALCULATED",
                                        pad=pads, data_mode=1, ceil_mode=ceil_mode,
                                        fusion_params=fusion_params)
        else:
            res = te.lang.cce.pooling2d(l1_width_fusion_in, window, stride, "AVG", padding,
                                        fusion_params=fusion_params)
    else:
        if padding == "CALCULATED":
            res = te.lang.cce.pooling2d(x, window, stride, "AVG", "CALCULATED", pad=pads, data_mode=1,
                                        ceil_mode=ceil_mode, fusion_params=fusion_params)
        else:
            res = te.lang.cce.pooling2d(x, window, stride, "AVG", padding,
                                        fusion_params=fusion_params)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_LIST_INT,
                 OPTION_ATTR_STR, OPTION_ATTR_LIST_INT, OPTION_ATTR_STR,
                 OPTION_ATTR_BOOL, OPTION_ATTR_BOOL, OPTION_ATTR_BOOL, KERNEL_NAME)
def avg_pool_v2(x, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                data_format="NCHW", global_pooling=False, ceil_mode=False,
                exclusive=True, kernel_name="avg_pool_v2"):
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

    kernel_name : cce kernel name, default value is "avg_pool_v2"

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
    avg_pool_v2_check_rule(input_shape, input_dtype, output_dtype,
                           input_format, ksize, strides, pads,
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

    # compute
    # create tensor_in
    tensor_in = tvm.placeholder(input_shape, name="tensor_in",
                                dtype=input_dtype, attrs=attr)

    res = avg_pool_v2_compute1(tensor_in, y, ksize, strides, padding, pads,
                               data_format, global_pooling, ceil_mode,
                               exclusive, False, kernel_name)

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
