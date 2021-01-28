#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import

from te import tvm
import te.lang.cce as tbe
import te.lang.base as tbe_base
from tbe.common.utils import para_check
from impl.util.util_cube_dynamic import Conv2dBackpropParaProcess
from impl.util.util_cube_dynamic import set_default_para

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3


def _conv2d_backprop_input_compute(input_size, filters, out_backprop, y, strides, pads,
                                   dilations=(1, 1, 1, 1), groups=1, data_format='NHWC',
                                   kernel_name='conv2d_backprop_input'): # pylint: disable=invalid-name, R0913
    ori_paras = {
        "input_size": input_size, "filters": filters, "out_backprop": out_backprop, "y": y,
        "strides": strides, "pads": pads, "dilations": dilations, "groups": groups, "data_format": data_format,
        "kernel_name": kernel_name
    }

    default_para = set_default_para()
    if not input_size.get("ori_shape"):
        ori_paras["input_size"]["ori_shape"] = default_para["input_size"]["ori_shape"]
    conv2dbp_para = Conv2dBackpropParaProcess(ori_paras)
    paras = conv2dbp_para.config_paras()

    dedx = tbe.conv2d_backprop_input_compute(
        filters=paras.get("filter_tensor"),
        out_backprop=paras.get("dy_tensor"),
        filter_sizes=paras.get("filter_shape"),
        input_sizes=paras.get("input_size"),
        para_dict={"strides": (conv2dbp_para.strides[H_DIM], conv2dbp_para.strides[W_DIM]),
                   "padding": conv2dbp_para.pads,
                   "dilations": conv2dbp_para.dilations,
                   "res_dtype": default_para.get("res_dtype"),
                   "kernel_name": kernel_name,
                   "group_dict": paras.get("group_para")})

    return {'op_placeholder': [paras.get("input_tensor"), paras.get("filter_tensor"), paras.get("dy_tensor")],
            'op_res': [dedx]}


@tbe_base.register_operator('Conv2DBackpropInput')
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (tuple, list), (tuple, list), int, str, str,
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

    pads: tuple/list of 4 integers
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

    with tbe_base.compute():
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
    tbe.build(sch, config)
