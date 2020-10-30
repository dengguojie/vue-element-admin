#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import

from te import tvm
import te.lang.cce as tbe
import te.lang.dynamic as dynamic
import te.platform as tbe_platform
from te.utils import check_para
from te.utils import error_manager
import impl.util.util_deconv_comm as comm


def _ceil(x_1, x_2):
    if x_2 == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        raise RuntimeError(dict_args,
                           error_manager.get_error_message(dict_args))
    return (x_1 + x_2 - 1) // x_2


def _get_pos_from_format(format_in):
    return format_in.find("N"), format_in.find("C"), format_in.find("H"), \
        format_in.find("W")


def _get_output(x_in, k_size, pads, stride, dilation):
    return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1


def _range_correction(fmap_range, kernel, padding, stride, dilation, out_shape):
    if padding == "VALID":
        padding = (0, 0, 0, 0)
    if padding == "SAME":
        out_h_lower = _ceil(fmap_range[2][0], stride[0])
        out_h_upper = _ceil(fmap_range[2][1], stride[0])
        out_w_lower = _ceil(fmap_range[3][0], stride[1])
        out_w_upper = _ceil(fmap_range[3][1], stride[1])
    else:
        out_h_lower = _get_output(fmap_range[2][0], kernel[2],
                                  (padding[0], padding[1]), stride[0], dilation[2])
        out_h_upper = _get_output(fmap_range[2][1], kernel[2],
                                  (padding[0], padding[1]), stride[0], dilation[2])
        out_w_lower = _get_output(fmap_range[3][0], kernel[3],
                                  (padding[2], padding[3]), stride[1], dilation[3])
        out_w_upper = _get_output(fmap_range[3][1], kernel[3],
                                  (padding[2], padding[3]), stride[1], dilation[3])
    return [(out_shape[0], out_shape[0]), (out_shape[1], out_shape[1]),
            (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]


def _check_and_config_para(filter, out_backprop, y, input_size, strides,
                           pads, dilations, data_format, kernel_name):
    ori_shape_filters = filter.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    range_y = y.get("range")
    filter_dtype = filter.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y.get("dtype")
    ori_format_y = y.get("ori_format")
    ori_format_filters = filter.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")

    if len(strides) == 4:
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[h_index], strides[w_index]]

    shape_filters = comm.get_filter_shape(
        ori_format_filters, ori_shape_filters)
    shape_out_backprop = comm.get_shape_out_backprop(
        ori_format_out_backprop, ori_shape_out_backprop)
    dilations = comm.get_shape_dilation(data_format, dilations)
    input_size = _get_input_size(y)
    dynamic_mode = _config_dynamic_para(shape_out_backprop)

    # get range
    if len(range_y) == 4:
        pos_n, pos_c, pos_h, pos_w = _get_pos_from_format(ori_format_y)
        range_y = [range_y[pos_n], range_y[pos_c], range_y[pos_h], range_y[pos_w]]
    elif len(range_y) == 5:
        range_y = [range_y[0], (input_size[1], input_size[1]),
                   range_y[2], range_y[3]]
        range_y = [tuple(r) for r in range_y]
    else:
        raise RuntimeError("range format should be same as input format")
    range_dedy = _range_correction(range_y, shape_filters, pads, strides,
                                   dilations, shape_out_backprop)

    dx_shape, dedy, filter_frac, input_size, shape_out_backprop = \
        _config_placeholder(
            shape_out_backprop, shape_filters, input_size,
            filter_dtype, out_backprop_dtype, range_dedy, range_y, dynamic_mode)

    shape_filter, shape_out_backprop, input_sizes, strides, pads, \
        dilations, filter_dtype, out_backprop_dtype, res_dtype, kernel_name \
        = comm.check_conv2dbp_input_params(shape_filters, shape_out_backprop,
                                           input_size, strides, pads,
                                           dilations, filter_dtype,
                                           out_backprop_dtype, res_dtype,
                                           kernel_name, None, dynamic_mode)

    return dx_shape, dedy, filter_frac, input_size, shape_filter, shape_out_backprop, \
        strides, pads, dilations, res_dtype


def _get_input_size(dx):
    shape = dx.get("ori_shape")
    format = dx.get("ori_format")
    idx_c, idx_h, idx_w = format.find("C"), format.find("H"), format.find("W")
    return [shape[0], shape[idx_c], shape[idx_h], shape[idx_w]]


def _config_dynamic_para(shape_dedy):
    if shape_dedy[2] == shape_dedy[3] == -1 and shape_dedy[0] != -1 and \
            shape_dedy[1] != -1:
        dynamic_mode = "dynamic_hw"
    elif shape_dedy[0] == -1 and -1 not in shape_dedy[1:]:
        dynamic_mode = "dynamic_batch"
    else:
        args_dict = {
            "errCode": "E60108",
            "op_name": "out_backprop",
            "reason": "only support dynamic_hw and dynamic_batch now."
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))

    return dynamic_mode


def _config_placeholder(shape_out_backprop, shape_filter, input_size,
                        out_backprop_dtype, filter_dtype, range_dy,
                        range_dx, dynamic_mode):
    _, dy_k0, _ = tbe_platform.CUBE_MKN[out_backprop_dtype]['mac']
    _, w_k0, w_n0 = tbe_platform.CUBE_MKN[filter_dtype]['mac']

    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_channel, filter_h, filter_w = shape_filter
    shape_filter_frac = (_ceil(filter_channel, w_n0)*filter_h*filter_w,
                         _ceil(filter_batch, w_k0), w_k0, w_n0)

    if dynamic_mode == "dynamic_hw":
        dedy_h = tbe_platform.var("dedy_h", range_dy[2])
        dedy_w = tbe_platform.var("dedy_w", range_dy[3])
        dx_h = tbe_platform.var("dx_h", range_dx[2])
        dx_w = tbe_platform.var("dx_w", range_dx[3])
        tbe_platform.add_exclude_bound_var(dedy_h)
        tbe_platform.add_exclude_bound_var(dedy_w)
        tbe_platform.add_exclude_bound_var(dx_h)
        tbe_platform.add_exclude_bound_var(dx_w)
        input_size[2] = dx_h
        input_size[3] = dx_w
    elif dynamic_mode == "dynamic_batch":
        dedy_batch = tbe_platform.var("batch_n", range_dx[0])
        tbe_platform.add_exclude_bound_var(dedy_batch)
        input_size[0] = dedy_batch

    shape_out_backprop = (dedy_batch, dedy_channel, dedy_h, dedy_w)
    shape_dedy = (dedy_batch,
                  _ceil(dedy_channel, dy_k0), dedy_h, dedy_w, dy_k0)

    dx_shape = tvm.placeholder([4], name="input_size", dtype="int32")
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    filter_frac = tvm.placeholder(shape_filter_frac,
                                  name="filter", dtype=filter_dtype)

    return dx_shape, dedy, filter_frac, input_size, shape_out_backprop


def _conv2d_backprop_input_compute(input_size, filter, out_backprop,
                                   y, strides, pads,
                                   dilations=(1, 1, 1, 1),
                                   groups=1, data_format='NHWC',
                                   kernel_name='conv2d_backprop_input'):
    dx_shape, dedy, filter_frac, input_size, shape_filter, shape_out_backprop, \
        strides, pads, dilations, res_dtype \
        = _check_and_config_para(filter, out_backprop, y, input_size, strides,
                                 pads, dilations, data_format, kernel_name)

    dedx = tbe.conv2d_backprop_input_compute(
        filters=filter_frac,
        out_backprop=dedy,
        filter_sizes=shape_filter,
        input_sizes=input_size,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype,
        kernel_name=kernel_name)

    return {'op_placeholder': [dx_shape, filter_frac, dedy],  'op_res': [dedx]}


@tbe_platform.register_operator('Conv2DBackpropInput')
@check_para.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (str, tuple, list), (tuple, list), int, str, str,
                             (type(None), dict))
def conv2d_backprop_input(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                          filter, out_backprop, y, strides,
                          pads, dilations=(1, 1, 1, 1),
                          groups=1, data_format="NHWC",
                          kernel_name="conv2d_backprop_input"):
    """
    algorithm: conv2d_backprop_input

    Parameters
    ----------
    input_size: dict, will not be used
            input tensor size.

    filter: dict with keys(ori_shape, ori_format, dtype)
            convolution kernel

    out_backprop: dict with keys(ori_shape, ori_format, dtype)
                  gradients.

    y: dict with keys(ori_shape, ori_format, dtype and range)
       conv2d_backprop_input output tensor

    strides: tuple/list of 4 integers
             filter move stride

    pads: string or tuple or list
          str: "SAME" or "VALID"
          tuple/list of 4 integers: [pad_top, pad_bottom, pad_left, pad_right]

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

    with tbe_platform.compute():
        res = _conv2d_backprop_input_compute(
            input_size, filter, out_backprop, y,
            strides, pads, dilations, groups, data_format, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get('op_res'))

    tensor_list = res.get('op_placeholder') + res.get('op_res')
    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensor_list,
              'build_args': {'constant_realize_extent_in_infer_bound': False}}
    dynamic.build(sch, config)
