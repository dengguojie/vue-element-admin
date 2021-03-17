#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_transpose
"""

from __future__ import absolute_import

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import Conv2dTransposeParaProcess
from impl.util.util_cube_dynamic import set_default_para


H_DIM = 2
W_DIM = 3


def _conv2d_transpose_compute(input_size, x, filter, bias, offset_w,
                              y, strides, pads,
                              dilations=(1, 1, 1, 1),
                              groups=1, data_format='NHWC', output_padding=(0, 0, 0, 0), offset_x=0,
                              kernel_name='conv2d_transpose'):

    ori_paras = {
        "input_size": input_size, "x": x, "filters": filter, "bias": bias, "offset_w": offset_w, "y": y,
        "strides": strides, "pads": pads, "dilations": dilations, "groups": groups, "data_format": data_format,
        "output_padding": output_padding, "offset_x": offset_x, "kernel_name": kernel_name
    }

    default_para = set_default_para()
    if not input_size.get("ori_shape"):
        ori_paras["input_size"]["ori_shape"] = default_para["input_size"]["ori_shape"]
    conv2dbp_para = Conv2dTransposeParaProcess(ori_paras)
    paras = conv2dbp_para.config_paras()

    dedx = tbe.conv2d_backprop_input(filters=paras.get("filter_tensor"),
                                     out_backprop=paras.get("x_tensor"),
                                     filter_sizes=paras.get("filter_shape"),
                                     input_sizes=paras.get("input_size"),
                                     para_dict={
                                         "strides":
                                         (conv2dbp_para.strides[H_DIM], conv2dbp_para.strides[W_DIM]),
                                         "padding": conv2dbp_para.pads,
                                         "dilations": conv2dbp_para.dilations,
                                         "res_dtype": default_para.get("res_dtype"),
                                         "tensor_bias": paras.get("bias_tensor"),
                                         "offset_x": offset_x,
                                         "kernel_name": kernel_name,
                                         "group_dict": paras.get("group_para")
                                     })

    if bias:
        return {'op_placeholder': [paras.get("input_tensor"), paras.get("x_tensor"),
                                   paras.get("filter_tensor"), paras.get("bias_tensor")],
                'op_res': [dedx]}
    return {'op_placeholder': [paras.get("input_tensor"), paras.get("x_tensor"), paras.get("filter_tensor")],
            'op_res': [dedx]}


@register_operator('Conv2DTranspose')
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

    with tbe.compute():
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
