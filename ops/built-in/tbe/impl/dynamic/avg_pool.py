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
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator

AVG_KERNEL_SIZE_H_MUL_W = 255 #kernel_h * kernel_w
AVG_KERNEL_SIZE = 20 # maximum ksize
MAX_CUBE_STRIDE = 63 # maximum cube stride
NONETYPE = type(None)

UNKNOWN_RANK_SHAPE = [-2]


def check_supported(x,
                    filter,
                    bias,
                    y,
                    ksize,
                    strides,
                    padding="VALID",
                    data_format="NHWC",
                    offset_x=0,
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
        reason = "data_format[%s] is not in [NHWC, NCHW]." % data_format
        return False, reason
    is_support_kernel = (ksize_h * ksize_w <= AVG_KERNEL_SIZE_H_MUL_W) or \
                        (ksize_h <= AVG_KERNEL_SIZE and ksize_w <= AVG_KERNEL_SIZE)
    if not is_support_kernel and outputh != 1 and outputw == 1:
        reason = "the shape is not supported by schedule, ksize:%s ori_shape:%s"\
                  % (str(ksize),str(ori_shape))
        return False, reason
    if not is_support_kernel and not (outputh == 1 and outputw == 1):
        reason = "the shape is not supported by schedule, ksize:%s ori_shape:%s"\
                  % (str(ksize),str(ori_shape))
        return False, reason
    return True, ""


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
        if padding == "SAME":
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
        error_manager_cube.raise_err_three_paras("E62304", "avg_pool", "ksize",
                                                 "4", str(len(ksize)))
    if len(strides) != 4:
        error_manager_cube.raise_err_three_paras("E62304", "avg_pool", "strides",
                                                 "4", str(len(strides)))

    dim_n, dim_c, dim_h, dim_w = 0, 1, 2, 3
    if data_format == "NHWC":
        dim_n, dim_h, dim_w, dim_c = 0, 1, 2, 3
    if ksize[dim_n] != 1 or ksize[dim_c] != 1:
        error_manager_cube.raise_err_three_paras("E62305", "avg_pool", "(ksize_n, ksize_c)",
                                                 "(1, 1)", str((ksize[dim_n], ksize[dim_c])))
    if ksize[dim_h] == -1 or ksize[dim_w] == -1:
        error_manager_cube.raise_err_specific_user("avg_pool",
                                                   "ksize_h/ksize_w should be const size.")
    if strides[dim_n] != 1 or strides[dim_c] != 1:
        error_manager_cube.raise_err_three_paras("E62305", "avg_pool", "(stride_n, stride_c)",
                                                 "(1, 1)", str((strides[dim_n], strides[dim_c])))
    if padding not in ("SAME", "VALID"):
        error_manager_cube.raise_err_three_paras("E62305", "avg_pool", "padding",
                                                 "SAME/VALID", padding)
    if data_format not in("NCHW", "NHWC"):
        error_manager_cube.raise_err_three_paras("E62305", "avg_pool", "data_format",
                                                 "NCHW/NHWC", data_format)
    if offset_x != 0:
        error_manager_cube.raise_err_three_paras("E62305", "avg_pool", "offset_x",
                                                 "0", str(offset_x))


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
        error_manager_cube.raise_err_specific_user("avg_pool",
                                                   "stride_h/stride_w should be less than " + \
                                                   str(MAX_CUBE_STRIDE) + " when filter is not None.")
    if filter_format not in ("NCHW", "NHWC"):
        error_manager_cube.raise_err_specific_user("avg_pool",
                                                   "ori_format of filter should be 'NCHW' or 'NHWC'.")
    h_index = filter_format.index("H")
    w_index = filter_format.index("W")
    c_index = filter_format.index("C")
    n_index = filter_format.index("N")
    if filter_shape[h_index] != window[0] or filter_shape[w_index] != window[1]:
        error_manager_cube.raise_err_three_paras("E62305", "avg_pool", "(filter_h, filter_w)",
                                                 str((window[0], window[1])),
                                                 str((filter_shape[h_index], filter_shape[w_index])))
    if filter_shape[c_index] != 1:
        error_manager_cube.raise_err_three_paras("E62305", "avg_pool", "filter_c",
                                                 "1", str(filter_shape[c_index]))
    if filter_shape[n_index] != fmap_shape[fmap.get("ori_format").index("C")]:
        error_manager_cube.raise_err_three_paras("E62305", "avg_pool", "filter_n",
                                                 "fmap_c", str(filter_shape[c_index]))


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
    input_shape = list(x.get("ori_shape"))
    input_dtype = x.get("dtype").lower()
    input_format = x.get("ori_format")
    output_dtype = y.get("dtype").lower()
    output_format = y.get("ori_format")
    if not output_format:
        y["ori_format"] = input_format
    elif output_format not in ("NCHW", "NHWC"):
        error_manager_cube.raise_err_one_para("E62006", "avg_pool",
                                              "output_format should be 'NCHW or 'NHWC'")

    if list(input_shape) == UNKNOWN_RANK_SHAPE:
        error_manager_cube.raise_err_specific_user("avg_pool",
                                                   "unknwon rank is not supported.")
    _avg_pool_check_rule(input_shape, input_dtype, output_dtype, ksize, strides, padding,
                         data_format, offset_x, kernel_name)
    if data_format == "NCHW":
        stride = [strides[2], strides[3]]
        window = [ksize[2], ksize[3]]
    elif data_format == "NHWC":
        stride = [strides[1], strides[2]]
        window = [ksize[1], ksize[2]]
    if input_format != data_format:
        error_manager_cube.raise_err_specific_user("avg_pool",
                                                   "ori_format of fmap should be equal with data_format.")
    if bias is None and filter is not None:
        dim_c = input_format.index("C")
        group = filter.get("ori_shape")[0]
        if input_shape[dim_c] == -1:
            error_manager_cube.raise_err_specific_user("avg_pool",
                                                       "dynamic c is not supported.")
        dilations = (1, 1, 1, 1)
        _check_filter_window(x, filter, window, stride)

        offset_w = None
        pad = padding

        conv2d(x, filter, bias, offset_w, y, strides, pad, dilations,
               groups=group, data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)
    else:
        if filter is None:
            error_manager_cube.raise_err_input_params_not_expected("dynamic_avg_pool", "filter", "dict",
                                                                   "None")
        if bias is not None:
            error_manager_cube.raise_err_input_params_not_expected("dynamic_avg_pool", "bias", "None",
                                                                   "dict")
