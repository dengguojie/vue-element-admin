#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import DepthwiseConv2dBackpropParaProcess
from impl.util.util_cube_dynamic import set_default_para


NONETYPE = type(None)
H_DIM = 2
W_DIM = 3


def _depthwise_conv2d_backprop_input_compute(
        input_size, filters, out_backprop, input_grad, strides, pads, dilations=(1, 1, 1, 1), data_format='NHWC',
        kernel_name='depthwise_conv2d_backprop_input'): # pylint: disable=invalid-name, R0913
    ori_paras = {
        "input_size": input_size, "filters": filters, "out_backprop": out_backprop, "input_grad": input_grad,
        "strides": strides, "pads": pads, "dilations": dilations, "data_format": data_format,
        "kernel_name": kernel_name
    }

    default_para = set_default_para()
    if not input_size.get("ori_shape"):
        ori_paras["input_size"]["ori_shape"] = default_para["input_size"]["ori_shape"]
    conv2dbp_para = DepthwiseConv2dBackpropParaProcess(ori_paras)
    paras = conv2dbp_para.config_paras()

    dedx = tbe.conv2d_backprop_input(
        filters=paras.get("filter_tensor"),
        out_backprop=paras.get("dy_tensor"),
        filter_sizes=paras.get("filter_shape"),
        input_sizes=paras.get("input_size"),
        para_dict={"strides": (conv2dbp_para.strides[H_DIM], conv2dbp_para.strides[W_DIM]),
                   "padding": conv2dbp_para.pads,
                   "dilations": conv2dbp_para.dilations,
                   "res_dtype": default_para.get("res_dtype"),
                   "kernel_name": kernel_name,
                   "group_dict": paras.get("group_para"),
                   "correct_range_flag": paras.get("correct_range_flag", False)})

    return {'op_placeholder': [paras.get("input_tensor"), paras.get("filter_tensor"), paras.get("dy_tensor")],
            'op_res': [dedx]}


@operation.register_operator('DepthwiseConv2DBackpropInput')
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (tuple, list), (tuple, list), str, str)
def depthwise_conv2d_backprop_input(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                    filter, out_backprop, input_grad, strides,
                                    dilations=(1, 1, 1, 1), pads=(0, 0, 0, 0), data_format="NHWC",
                                    kernel_name="depthwise_conv2d_backprop_input"):
    """
    algorithm: depthwise_conv2d_backprop_input

    Parameters
    ----------
    input_size: dict, shape of input tensor, support [N, C, H, W] or [N, H, W, C], will not be used.

    filter: dict
        4-D origin shape and dtype of filter tensor
        support [H, W, C, K], K is channel_multiplier

    out_backprop: dict
        4-D origin shape and dtype of out_backprop tensor,
        support [N, Co, Ho, Wo] or [N, Ho, Wo, Co],
        gradients w.r.t. the output of the convolution

    input_grad: dict
        4-D origin shape and dtype of input tensor,
        support [N, C, H, W] or [N, H, W, C]

    strides: a list or tuple of four ints
        the stride of the sliding window for height and width of the input of
        the convolution, support [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations: an optional list or tuple of four ints
        the dilation factor for each dimension of input
        if set to k > 1, there will be k-1 skipped cells between each
        filter element on that dimension, support [1, 1, dilation_height,
        dilation_width] or [1, dilation_height, dilation_width, 1]

    pads: a list or tuple of four ints
        padding added to each dimension of the input

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    kernel_name: str
        cce kernel name, default value is "depthwise_conv2d_backprop_input"

    Returns
    -------
    None
    """

    with tbe.compute():
        res = _depthwise_conv2d_backprop_input_compute(
            input_size, filter, out_backprop, input_grad,
            strides, pads, dilations, data_format, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get('op_res'))

    tensor_list = res.get('op_placeholder') + res.get('op_res')
    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensor_list,
              'build_args': {'constant_realize_extent_in_infer_bound': False}}
    tbe.build(sch, config)