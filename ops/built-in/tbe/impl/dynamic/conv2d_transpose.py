#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_transpose
"""

from __future__ import absolute_import

from te import tvm
import te.lang.cce as tbe
import te.platform as tbe_platform
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils import error_manager
from te.utils.error_manager import error_manager_util
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
    pos_dict = {
        "pos_n": format_in.find("N"),
        "pos_c": format_in.find("C"),
        "pos_h": format_in.find("H"),
        "pos_w": format_in.find("W"),
    }
    return pos_dict


def _get_output(x_in, k_size, pads, stride, dilation):
    return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1


def _range_correction(fmap_range, kernel, pads, stride, dilation, out_shape):
    if -1 in pads:
        out_h_lower = _ceil(fmap_range[2][0], stride[0])
        out_h_upper = _ceil(fmap_range[2][1], stride[0])
        out_w_lower = _ceil(fmap_range[3][0], stride[1])
        out_w_upper = _ceil(fmap_range[3][1], stride[1])
    else:
        out_h_lower = _get_output(fmap_range[2][0], kernel[2],
                                  (pads[0], pads[1]), stride[0], dilation[2])
        out_h_upper = _get_output(fmap_range[2][1], kernel[2],
                                  (pads[0], pads[1]), stride[0], dilation[2])
        out_w_lower = _get_output(fmap_range[3][0], kernel[3],
                                  (pads[2], pads[3]), stride[1], dilation[3])
        out_w_upper = _get_output(fmap_range[3][1], kernel[3],
                                  (pads[2], pads[3]), stride[1], dilation[3])
    return [(out_shape[0], out_shape[0]), (out_shape[1], out_shape[1]),
            (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]


def _check_and_config_para(input_size, x, filter, bias, offset_w, y, strides,
                           pads, dilations, groups, data_format, output_padding, offset_x, kernel_name):
    ori_shape_filters = filter.get("ori_shape")
    filter_dtype = filter.get("dtype")
    ori_format_filters = filter.get("ori_format")

    ori_shape_x = x.get("ori_shape")
    x_dtype = x.get("dtype")
    ori_format_x = x.get("ori_format")

    range_y = y.get("range")
    res_dtype = y.get("dtype")
    ori_format_y = y.get("ori_format")

    if len(strides) == 4:
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[h_index], strides[w_index]]

    shape_filters = comm.get_filter_shape(ori_format_filters, ori_shape_filters)
    shape_x = comm.get_shape_out_backprop(ori_format_x, ori_shape_x)
    dilations = comm.get_shape_dilation(data_format, dilations)
    input_size = _get_input_size(y)
    dynamic_mode = _config_dynamic_para(shape_x)
    bias_flag = bias is not None

    # get range
    if len(range_y) == 4:
        pos_dict = _get_pos_from_format(ori_format_y)
        pos_n = pos_dict["pos_n"]
        pos_c = pos_dict["pos_c"]
        pos_h = pos_dict["pos_h"]
        pos_w = pos_dict["pos_w"]
        range_y = [range_y[pos_n], range_y[pos_c], range_y[pos_h], range_y[pos_w]]
    elif len(range_y) == 5:
        range_y = [range_y[0], (input_size[1], input_size[1]),
                   range_y[2], range_y[3]]
        range_y = [tuple(r) for r in range_y]
    else:
        raise RuntimeError("range format should be same as input format")
    range_dedy = _range_correction(range_y, shape_filters, pads, strides,
                                   dilations, shape_x)


    placeholder_dict = \
        _config_placeholder(
            shape_x, shape_filters, input_size,
            filter_dtype, x_dtype, range_dedy, range_y, dynamic_mode)
    dx_shape = placeholder_dict["dx_shape"]
    dedy = placeholder_dict["dedy"]
    filter_frac = placeholder_dict["filter_frac"]
    input_size = placeholder_dict["input_size"]
    shape_x = placeholder_dict["shape_x"]

    if groups != 1:
        dict_args = {"errCode": "E60108", "reason": "group must be 1 now"}
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args)
        )

    group_dict = comm.calculate_group(
        shape_x,
        input_size,
        shape_filters,
        groups,
        filter_dtype,
        ori_format_filters,
    )
    shape_filter, shape_x, input_size, strides, pads, \
    dilations, filter_dtype, x_dtype, res_dtype, kernel_name \
        = comm.check_conv2dbp_input_params(shape_filters, shape_x,
                                           input_size, strides, pads,
                                           dilations, filter_dtype,
                                           x_dtype, res_dtype,
                                           kernel_name,
                                           dynamic_mode=dynamic_mode,
                                           group_dict=group_dict)


    _, dedy_channel, _, _ = shape_x
    _, _, w_n0 = tbe_platform.CUBE_MKN[filter_dtype]["mac"]
    if bias_flag:
        input_channel = comm.align(dedy_channel, w_n0)
        tensor_bias = tvm.placeholder(
            (input_channel,), name="tensor_bias", dtype=res_dtype
        )
    else:
        tensor_bias = None
    check_and_config_para_dict = {
        "dx_shape": dx_shape,
        "dedy": dedy,
        "filter_frac": filter_frac,
        "input_size": input_size,
        "shape_filter": shape_filter,
        "shape_x": shape_x,
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "group_dict": group_dict,
        "tensor_bias": tensor_bias,
    }

    return check_and_config_para_dict


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


def _config_placeholder(shape_x, shape_filter, input_size,
                        x_dtype, filter_dtype, range_dy,
                        range_dx, dynamic_mode):
    _, dy_k0, _ = tbe_platform.CUBE_MKN[x_dtype]['mac']
    _, w_k0, w_n0 = tbe_platform.CUBE_MKN[filter_dtype]['mac']

    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_x
    filter_batch, filter_channel, filter_h, filter_w = shape_filter
    shape_filter_frac = (_ceil(filter_channel, w_n0) * filter_h * filter_w,
                         _ceil(filter_batch, w_k0), w_k0, w_n0)

    if dynamic_mode == "dynamic_hw":
        dedy_h = tbe_base.var("dedy_h", range_dy[2])
        dedy_w = tbe_base.var("dedy_w", range_dy[3])
        dx_h = tbe_base.var("dx_h", range_dx[2])
        dx_w = tbe_base.var("dx_w", range_dx[3])
        tbe_base.add_exclude_bound_var(dedy_h)
        tbe_base.add_exclude_bound_var(dedy_w)
        tbe_base.add_exclude_bound_var(dx_h)
        tbe_base.add_exclude_bound_var(dx_w)
        input_size[2] = dx_h
        input_size[3] = dx_w
    elif dynamic_mode == "dynamic_batch":
        dedy_batch = tbe_base.var("batch_n", range_dx[0])
        tbe_base.add_exclude_bound_var(dedy_batch)
        input_size[0] = dedy_batch

    shape_x = (dedy_batch, dedy_channel, dedy_h, dedy_w)
    shape_dedy = (dedy_batch,
                  _ceil(dedy_channel, dy_k0), dedy_h, dedy_w, dy_k0)

    dx_shape = tvm.placeholder([4], name="input_size", dtype="int32")
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=x_dtype)
    filter_frac = tvm.placeholder(shape_filter_frac,
                                  name="filter", dtype=filter_dtype)

    placeholder_dict = {
        "dx_shape": dx_shape,
        "dedy": dedy,
        "filter_frac": filter_frac,
        "input_size": input_size,
        "shape_x": shape_x,
    }
    return placeholder_dict


def _conv2d_transpose_compute(input_size, x, filter, bias, offset_w,
                              y, strides, pads,
                              dilations=(1, 1, 1, 1),
                              groups=1, data_format='NHWC', output_padding=(0, 0, 0, 0), offset_x=0,
                              kernel_name='conv2d_transpose'):
    check_and_config_para_dict = _check_and_config_para(input_size, x, filter, bias, offset_w, y, strides,
                                 pads, dilations, groups, data_format, output_padding, offset_x, kernel_name)
    dx_shape = check_and_config_para_dict["dx_shape"]
    dedy = check_and_config_para_dict["dedy"]
    filter_frac = check_and_config_para_dict["filter_frac"]
    input_size = check_and_config_para_dict["input_size"]
    shape_filter = check_and_config_para_dict["shape_filter"]
    shape_x = check_and_config_para_dict["shape_x"]
    strides = check_and_config_para_dict["strides"]
    pads = check_and_config_para_dict["pads"]
    dilations = check_and_config_para_dict["dilations"]
    res_dtype = check_and_config_para_dict["res_dtype"]
    group_dict = check_and_config_para_dict["group_dict"]
    tensor_bias = check_and_config_para_dict["tensor_bias"]
    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "tensor_bias": tensor_bias,
        "offset_x": offset_x,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    dedx = tbe.conv2d_backprop_input_compute(
        filters=filter_frac,
        out_backprop=dedy,
        filter_sizes=shape_filter,
        input_sizes=input_size,
        para_dict=para_dict
    )
    if bias:
        return {'op_placeholder': [dx_shape, dedy, filter_frac, tensor_bias], 'op_res': [dedx]}
    else:
        return {'op_placeholder': [dx_shape, dedy, filter_frac], 'op_res': [dedx]}


@tbe_base.register_operator('Conv2DTranspose')
@para_check.check_input_type(dict, dict, dict, (type(None), dict), (type(None), dict), dict, (tuple, list),
                             (tuple, list), (tuple, list), int, str, (tuple, list), int, str,
                             (type(None), dict))
def conv2d_transpose(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                     x, filter, bias, offset_w, y, strides,
                     pads, dilations=(1, 1, 1, 1),
                     groups=1, data_format="NHWC", output_padding=(0, 0, 0, 0), offset_x=0,
                     kernel_name="conv2d_transpose"):
    """
    algorithm: conv2d_transpose

    Parameters
    ----------
    input_size: dict, will not be used
            input tensor size.

    x: dict with keys(ori_shape, ori_format, dtype)
        The shape of gradients.

    filter: dict with keys(ori_shape, ori_format, dtype)
            convolution kernel

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
        Input offset_w tensor.

    y: dict with keys(ori_shape, ori_format, dtype and range)
       conv2d_transpose output tensor

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          str: "SAME" or "VALID"
          tuple/list of 4 integers: [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_transpose
    groups: int
            param for group conv2d_transpose

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    output_padding: tuple/list of 4 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "conv2d_transpose"

    Returns
    -------
    None
    """

    with tbe_base.compute():
        res = _conv2d_transpose_compute(
            input_size, x, filter, bias, offset_w, y,
            strides, pads, dilations, groups, data_format, output_padding, offset_x, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get('op_res'))

    tensor_list = res.get('op_placeholder') + res.get('op_res')
    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensor_list,
              'build_args': {'constant_realize_extent_in_infer_bound': False}}
    tbe.build(sch, config)
