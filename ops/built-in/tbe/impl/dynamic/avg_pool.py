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
avg_pool
"""

from impl.dynamic.conv2d import conv2d
from impl.util.platform_adapter import para_check
from te.utils.error_manager import error_manager_cube as err_man_cube
from impl.util import util_select_op_base
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

AVG_KERNEL_SIZE_H_MUL_W = 255 #kernel_h * kernel_w
AVG_KERNEL_SIZE = 20 # maximum ksize
MAX_CUBE_STRIDE = 63 # maximum cube stride
NONETYPE = type(None)


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=invalid-name,redefined-builtin,too-many-locals,unused-argument,unused-variable,unnecessary-lambda
def check_supported(x, filter, bias, y, ksize, strides,
                    padding="VALID", data_format="NHWC", offset_x=0,
                    kernel_name="avg_pool"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    True or False
    """
    ori_shape = y.get("ori_shape")
    if data_format == "NHWC":
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        outputh = ori_shape[1]
        outputw = ori_shape[2]
    elif data_format == "NCHW":
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        outputh = ori_shape[2]
        outputw = ori_shape[3]
    else:
        return False
    is_support_kernel = (ksize_h * ksize_w <= AVG_KERNEL_SIZE_H_MUL_W) or \
                        (ksize_h <= AVG_KERNEL_SIZE and ksize_w <= AVG_KERNEL_SIZE)
    if not is_support_kernel and outputh != 1 and outputw == 1:
        return False
    if not is_support_kernel and not (outputh == 1 and outputw == 1):
        return False
    return True


def get_op_support_info(x, filter, bias, y, ksize, strides,
                        padding="VALID", data_format="NHWC", offset_x=0,
                        kernel_name="avg_pool"):
    """
    get the avgpool split
    """
    format_x = x.get("format")
    input_shape = x.get("shape")

    if data_format in ("NHWC",):
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        window = [ksize[1], ksize[2]]
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        window = [ksize[2], ksize[3]]

    if format_x == "NC1HWC0":
        if (ksize_h == window[0] and ksize_w == window[1]) or padding == "SAME":
            axis_split_matrix = [[util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
                                 util_select_op_base.SplitOutput([0, [0]])]]
        elif padding == "VALID":
            axis_split_matrix = [
                [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                [util_select_op_base.SplitInput([0, [2], [0], [0]]), util_select_op_base.SplitOutput([0, [2]])]]
        else:
            axis_split_matrix = None
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list, 2, 0)

    return op_cal_info_in_json


# pylint: disable=locally-disabled,too-many-arguments,too-many-statements
# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _check_window_rule(ksize, strides, padding, data_format, offset_x):
    """
    check ksize and strides of window in pooling
    :param ksize: list or tuple, the length must be 4
    :param strides: list or tuple, the length must be 4
    :param data_format: input format
    :return: None
    """
    if len(ksize) != 4:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'ksize'
        error_info['min_value'] = '4'
        error_info['max_value'] = '4'
        error_info['real_value'] = len(ksize)
        raise RuntimeError(error_info,
                           "In op[%s], the num of dimensions of input[%s] "
                           "should be in the range of [%s, %s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['real_value']))

    if len(strides) != 4:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'strides'
        error_info['min_value'] = '4'
        error_info['max_value'] = '4'
        error_info['real_value'] = len(strides)
        raise RuntimeError(error_info,
                           "In op[%s], the num of dimensions of input[%s] "
                           "should be in the range of [%s, %s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['real_value']))

    ksize_c = ksize[3] if data_format in ("NHWC",) else ksize[1]
    strides_c = strides[3] if data_format in ("NHWC",) else strides[1]
    if ksize[0] != 1 or (ksize_c != 1):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = ",".join(("ksize[1]", "ksize[3]"))
        error_info['expected_value'] = '1'
        error_info['real_value'] = ",".join((str(ksize[1]), str(ksize[3])))
        raise RuntimeError("In op[%s], the parameter[%s] should be [%s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['expected_value'],
                            error_info['real_value']))


    if strides[0] != 1 or strides_c != 1:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = ",".join(("strides[1]", "strodes[3]"))
        error_info['expected_value'] = '1'
        error_info['real_value'] = ",".join((str(strides[1]), str(strides[3])))
        raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be [%s], "
                           "but actually is [%s]." % (error_info['op_name'],
                                                      error_info['param_name'],
                                                      error_info['expected_value'],
                                                      error_info['real_value']))

    if padding not in ("SAME", "VALID"):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_015
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'padding'
        error_info['expected_value_list'] = ",".join(("SAME", "VALID"))
        error_info['real_value'] = padding
        raise RuntimeError(error_info, "In op[%s], parameter[%s] should be one of [%s], "
                            "but actually is [%s]." % (error_info['op_name'],
                                                       error_info['param_name'],
                                                       error_info['expected_value_list'],
                                                       error_info['real_value']))

    if data_format not in("NCHW", "NHWC", "NC1HWC0"):
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_015
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'x'
        error_info['excepted_format_list'] = ",".join(("NC1HWC0",
                                                       "NCHW", "NHWC"))
        error_info['format'] = data_format
        raise RuntimeError(error_info, "In op[%s], the format[%s] of input "
                                       "should be one of [%s], "
                                       "but actuall"
                                       "y is [%s]."
                           % (error_info['op_name'],
                              error_info['param_name'],
                              error_info['excepted_format_list'],
                              error_info['format']))

    if offset_x != 0:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'offset_x'
        error_info['expected_value'] = '0'
        error_info['real_value'] = str(offset_x)
        raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be [%s], "
                                       "but actually is [%s]."
                           % (error_info['op_name'],
                              error_info['param_name'],
                              error_info['expected_value'],
                              error_info['real_value']))


def _avg_pool_check_rule(input_shape, input_dtype, output_dtype,
                          ksize, strides, padding, data_format, offset_x, kernel_name):
    """
    :param input_shape: shape of input_data
    :param input_dtype: dtype of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of avgpooling
    :param strides: the stride of avgpooling window
    :param padding: padding_mode of avgpool
    :param data_format: NHWC default
    :param offset_x: default 0
    :param kernel_name: cce kernel name
    :return: None
    """

    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ["float16", "int8"])
    para_check.check_dtype(output_dtype, ["float16", "int8", "int32"])

    _check_window_rule(ksize, strides, padding, data_format, offset_x)


def _check_filter_window(fmap, filter, window, stride):
    fmap_shape = fmap.get("ori_shape")
    filter_shape = filter.get("ori_shape")
    filter_format = filter.get("ori_format")
    if stride[0] > MAX_CUBE_STRIDE or stride[1] > MAX_CUBE_STRIDE:
        raise RuntimeError("In op[%s], the [%s] should less than [%s] when filter is None"
                           % ('avgpool', 'stride', str(MAX_CUBE_STRIDE)))
    if filter_format not in ("NCHW", "NHWC"):
        raise RuntimeError("In op[%s], the ori_format of filter"
                                       "should be [%s] or [%s]"
                           % ('avgpool', 'NCHW', 'NHWC'))
    h_index = filter_format.index("H")
    w_index = filter_format.index("W")
    c_index = filter_format.index("C")
    n_index = filter_format.index("N")
    ksize_h = filter_shape[h_index]
    ksize_w = filter_shape[w_index]
    if ksize_h * ksize_w > AVG_KERNEL_SIZE_H_MUL_W:
        raise RuntimeError("In op[%s], the ksize_h * ksize_w"
                           "should be less than [%s],"
                           % ('avgpool', str(AVG_KERNEL_SIZE_H_MUL_W)))
    if ksize_h > AVG_KERNEL_SIZE or ksize_w > AVG_KERNEL_SIZE:
        raise RuntimeError("In op[%s], the ksize_h and ksize_w"
                           "should be less than [%s],"
                           % ('avgpool', str(AVG_KERNEL_SIZE)))
    if filter_shape[h_index] != window[0] or filter_shape[w_index] != window[1]:
        raise RuntimeError("In op[%s], the h_shape of filter"
                                       "should be equal with [%s],"
                           % ('avgpool', 'ksize'))
    if filter_shape[c_index] != 1:
        raise RuntimeError("In op[%s], the c_shape of filter"
                                       "should be [%s],"
                           % ('avgpool', '1'))
    if filter_shape[n_index] != fmap_shape[fmap.get("ori_format").index("C")]:
        raise RuntimeError("In op[%s], the N shape of filter"
                                       "should be equal with C shape of fmap,"
                           % ('avgpool'))


@register_operator("AvgPool")
@para_check.check_input_type(dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list),
                             str, str, int, str)
def avg_pool(x, filter, bias, y, ksize, strides,
             padding="VALID", data_format="NHWC", offset_x=0,
             kernel_name="avg_pool"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    None
    """

    # get shape&dtype
    # input_shape only support format NCHW
    input_shape = x.get("ori_shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    input_format = x.get("ori_format")
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    if input_format == "NCHW":
        input_c, input_h, input_w = input_shape[1:4]
        stride = [-1, -1, strides[2], strides[3]]
        window = [ksize[2], ksize[3]]
    elif input_format == "NHWC":
        input_h, input_w, input_c = input_shape[1:4]
        stride = [-1, strides[1], strides[2], -1]
        window = [ksize[1], ksize[2]]
    else:
        raise RuntimeError("Unsupported input format!")

    _avg_pool_check_rule(input_shape, input_dtype, output_dtype, ksize, strides, padding,
                         data_format, offset_x, kernel_name)

    tbe_context.get_context().add_compile_info("strideh", stride[0])
    tbe_context.get_context().add_compile_info("stridew", stride[1])

    if bias is None and filter is not None:
        dilations = (1, 1, 1, 1)
        _check_filter_window(x, filter, window, stride)

        offset_w = None
        pad = padding

        conv2d(x, filter, bias, offset_w, y, stride, pad, dilations,
               groups=input_c, data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)
    else:
        if filter is None:
            err_man_cube.raise_err_input_params_not_expected("dynamic_avg_pool",
                                                             "filter",
                                                             "dict",
                                                             "None")
        if bias is not None:
            err_man_cube.raise_err_input_params_not_expected("dynamic_avg_pool",
                                                             "bias",
                                                             "None",
                                                             "dict")
