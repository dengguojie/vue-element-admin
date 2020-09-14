#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import
from collections import OrderedDict

import te.lang.dynamic
import te.lang.cce
from te import tvm
from topi import generic
from te.platform import CUBE_MKN
from te.platform import operation
from te import platform as cceconf
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import dynamic_build_config, build_config_update, build_config
from te.utils.error_manager import error_manager_util as err_man
from topi.cce import util
import impl.util.util_deconv_comm as comm

# the dim of shape in conv2d_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4
# the dim of strides in conv2d_backprop must be 2
STRIDES_SHAPE_DIM = 2
# the dim of pads in conv2d_backprop must be 4
PADDING_SHAPE_DIM = 4

# fmap_H, fmap_W must be in [2,4096]
FMAP_HW_MIN = 2
FMAP_HW_MAX = 4096

# DeDy_H,DeDy_W must be in [2,4096]
DEDY_HW_MIN = 2
DEDY_HW_MAX = 4096

# filter_H, filter_W must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

# stride must be in [1,63] and h*w can't larger than 256
STRIDE_HW_MIN = 1
STRIDE_HW_MAX = 63
STRIDE_SIZE_MAX = 256

# pads must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# each axis of shape must less than 1000000
DEFAULT_MAX_SHAPE_NUM = 1000000

# the bytes length of several dtypes
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
# same as (2**63-1)
DATA_SIZE_MAX = 9223372036854775807

# If pads is string , only support "SAME" or "VALID"
PADDING_SUPPORT = ('SAME', 'VALID')
# pads valid mode is [0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0]


def _ceil(x_1, x_2):
    if x_2 == 0:
        raise RuntimeError("Division by zero")
    return (x_1 + x_2 - 1) // x_2


def check_and_config_para(filter, out_backprop, y, input_size, strides,
                          pads, dilations, data_format, kernel_name):
    ori_shape_filters = filter.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_range_filters = filter.get("range")
    ori_range_out_backprop = out_backprop.get("range")
    ori_range_y = y.get("range")
    filter_dtype = filter.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y.get("dtype")
    ori_format_filters = filter.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")

    if len(strides) == 4:
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[h_index], strides[w_index]]

    shape_filters = comm.get_filter_shape(
        ori_format_filters, ori_shape_filters
    )
    shape_out_backprop = comm.get_shape_out_backprop(
        ori_format_out_backprop, ori_shape_out_backprop)
    dilations = comm.get_shape_dilation(data_format, dilations)

    dynamic_mode = config_dynamic_para(shape_out_backprop)
    dedy, filter_frac, input_size, shape_out_backprop, var_map = config_placeholder(\
        shape_out_backprop, shape_filters, input_size, \
        filter_dtype, out_backprop_dtype, \
        ori_range_filters, ori_range_out_backprop, ori_range_y, \
        dynamic_mode)

    shape_filter, shape_out_backprop, input_sizes, strides, pads, \
        dilations, filter_dtype, out_backprop_dtype, res_dtype, kernel_name \
         = comm.check_conv2dbp_input_params(shape_filters, shape_out_backprop,
                                           input_size, strides, pads,
                                           dilations, filter_dtype,
                                           out_backprop_dtype, res_dtype,
                                           kernel_name, None, dynamic_mode)

    return dedy, filter_frac, input_size, shape_filter, shape_out_backprop, \
           strides, pads, dilations, res_dtype, var_map, dynamic_mode, \
               ori_range_out_backprop, ori_range_y


def config_dynamic_para(shape_dedy):
    if shape_dedy[2] == shape_dedy[3] == -1 and shape_dedy[0] != -1 and shape_dedy[1] != -1:
        dynamic_mode = "dynamic_hw"
    elif shape_dedy[0] == -1 and -1 not in shape_dedy[1:]:
        dynamic_mode = "dynamic_batch"
    else:
        args_dict = {
            "errCode": "E60108",
            "op_name": "out_backprop",
            "reason": "only support dynamic_hw and dynamic_batch now."
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    return dynamic_mode


def config_placeholder(shape_out_backprop, shape_filter, input_size, \
                       out_backprop_dtype, filter_dtype, \
                       ori_range_filters, ori_range_out_backprop, \
                       ori_range_y, dynamic_mode):
    _, dy_k0, _ = CUBE_MKN[out_backprop_dtype]['mac']
    _, w_k0, w_n0 = CUBE_MKN[filter_dtype]['mac']

    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_channel, filter_h, filter_w = shape_filter
    shape_filter_frac = (
        _ceil(filter_channel, w_n0)*filter_h*filter_w,
        _ceil(filter_batch, w_k0), w_k0, w_n0)

    var_map = {}
    if dynamic_mode == "dynamic_hw":
        var_map["dedy_h"] = operation.var("dedy_h", tuple(ori_range_out_backprop[2]))
        var_map["dedy_w"] = operation.var("dedy_w", tuple(ori_range_out_backprop[3]))
        var_map["dx_h"] = operation.var("dx_h", tuple(ori_range_y[2]))
        var_map["dx_w"] = operation.var("dx_w", tuple(ori_range_y[3]))
        operation.add_exclude_bound_var(var_map["dedy_h"])
        operation.add_exclude_bound_var(var_map["dedy_w"])
        operation.add_exclude_bound_var(var_map["dx_h"])
        operation.add_exclude_bound_var(var_map["dx_w"])
        dedy_h = var_map["dedy_h"]
        dedy_w = var_map["dedy_w"]
        input_size[2] = var_map["dx_h"]
        input_size[3] = var_map["dx_w"]
    elif dynamic_mode == "dynamic_batch":
        var_map['batch_n'] = operation.var("batch_n", tuple(ori_range_out_backprop[0]))
        operation.add_exclude_bound_var(var_map['batch_n'])
        dedy_batch = var_map['batch_n']
        input_size[0] = var_map['batch_n']

    shape_out_backprop = (dedy_batch, dedy_channel, dedy_h, dedy_w)
    shape_dedy = (dedy_batch,
                  _ceil(dedy_channel, dy_k0), dedy_h, dedy_w, dy_k0)

    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    filter_frac = tvm.placeholder(shape_filter_frac,
                                  name="filter", dtype=filter_dtype)

    return dedy, filter_frac, input_size, shape_out_backprop, var_map


def conv2d_backprop_input_d_compute(filter, out_backprop, y,
                                    input_size, strides,
                                    pads, dilations=(1, 1, 1, 1),
                                    groups=1, data_format='NHWC',
                                    kernel_name='conv2d_backprop_input'):
    dedy, filter_frac, input_size, shape_filter, shape_out_backprop, \
           strides, pads, dilations, res_dtype, var_map, dynamic_mode, \
               ori_range_out_backprop, ori_range_y \
        = check_and_config_para(filter, out_backprop, y, input_size, strides,
                            pads, dilations, data_format, kernel_name)

    dedx = te.lang.cce.conv2d_backprop_input_compute(
        filters=filter_frac,
        out_backprop=dedy,
        filter_sizes=shape_filter,
        input_sizes=input_size,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype,
        dynamic_para={
            'dynamic_mode':dynamic_mode,
            'var_map':var_map
        },
        kernel_name=kernel_name)

    te.op.add_compile_info('dynamic_mode', dynamic_mode)

    return {'op_placeholder':[filter_frac, dedy],  'op_res':[dedx]}


@te.op.register_operator('Conv2DBackpropInputD')
@util.check_input_type(dict, dict, dict, (tuple, list), (tuple, list),
                       (str, tuple, list), (tuple, list), int, str, str,
                       (type(None), dict))
def conv2d_backprop_input_d(filter,  # pylint: disable=W0622,C0103,R0913,R0914
                            out_backprop, y, input_size, strides,
                            pads, dilations=(1, 1, 1, 1),
                            groups=1, data_format="NHWC",
                            kernel_name="conv2d_backprop_input"):
    """
    algorithm: conv2d_backprop_input

    Parameters
    ----------
    filter: dict with keys(shape, dtype and range)
            convolution kernel

    out_backprop: dict with keys(shape, dtype and range)
                  gradients.

    y: dict with keys(shape, dtype and range)
       conv2d_backprop_input output tensor, dtype must be assigned

    input_size: The shape of feature map.
                 4-D with shape [batch, channels, height, weight].

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_backprop_input
    groups: int
            param for group conv2d_backprop_input

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "conv2d_backprop_input"

    Returns
    -------
    None
    """

    with te.op.compute():
        res = conv2d_backprop_input_d_compute(filter, out_backprop, y, list(input_size),
            strides, pads, dilations, groups, data_format, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res.get('op_res'))

    tensor_list = res.get('op_placeholder') + res.get('op_res')
    config = {'print_ir':False,
        'name':kernel_name,
        'tensor_list':tensor_list,
        'build_args':{'constant_realize_extent_in_infer_bound': False}}
    te.lang.dynamic.build(sch, config)
