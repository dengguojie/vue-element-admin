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
from impl.util.platform_adapter import error_manager_cube as err_man
from impl.util.platform_adapter import tbe_register
from impl.util.util_cube_dynamic import DepthwiseConv2dBackpropParaProcess
from impl.util.util_cube_dynamic import Conv2dTransposeParaProcess
from impl.util.util_cube_dynamic import set_default_para
from impl.util.util_cube_dynamic import modify_w_range_max

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3
ORI_SHAPE_LEN = 4
SHAPE_LEN = 5


@tbe_register.register_param_generalization("DepthwiseConv2DBackpropInput")
def depthwise_conv2d_backprop_input_generalization(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                                   filter, out_backprop, input_grad, strides,
                                                   dilations=(1, 1, 1, 1), pads=(0, 0, 0, 0), data_format="NHWC",
                                                   kernel_name="depthwise_conv2d_backprop_input",
                                                   generalize_config={"mode": "keep_rank"}):
    """
    depthwise conv2d backprop input generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to depthwise_conv2d_backprop_input

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    support_mode = ["keep_rank"]
    if generalize_config["mode"] not in support_mode:
        err_man.raise_err_specific_user("depthwise_conv2d_backprop_input",
                                        "invalid generalize mode {}, only support {}".format(
                                            str(generalize_config["mode"]), str(support_mode)))
    result = []
    if generalize_config["mode"] == "keep_rank":  # fuzz build situation
        # unknow_rank inputs ori_shape is [-2], others' shape length is 4
        unknow_rank = len(out_backprop["ori_shape"]) == 1 and out_backprop["ori_shape"][0] == -2
        if unknow_rank:
            err_man.raise_err_specific_user("depthwise_conv2d_backprop_input",
                                            "not support unknow_rank under mode {}".format(
                                                generalize_config["mode"]))
        have_range = {"inputs": out_backprop, "outputs": input_grad}
        support_format = ["NCHW", "NHWC"]
        for name, tensor in have_range.items():
            if tensor.get("ori_format") not in support_format:
                err_man.raise_err_specific_user("depthwise_conv2d_backprop_input",
                                                "invalid {} ori_format {}, only support {}".format(
                                                    name, str(tensor.get("ori_format")), str(support_format)))
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                err_man.raise_err_specific_user("depthwise_conv2d_backprop_input",
                                                "invalid {} ori_shape {}, only support {}d".format(
                                                    name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
            valid = isinstance(tensor.get("shape"), (list, tuple)) and len(tensor["shape"]) == SHAPE_LEN
            if not valid:
                err_man.raise_err_specific_user("depthwise_conv2d_backprop_input",
                                                "invalid {} ori_shape {}, only support {}d".format(
                                                    name, str(tensor.get("shape")), str(SHAPE_LEN)))
        # if over l1 size then modify w range
        dy_w_range_max, is_single_point = modify_w_range_max(
            input_grad,
            filter,
            out_backprop,
            strides,
            data_format,
            "depthwise_conv2d_backprop_input")

        # get dx_range depends on dy_range
        dy_range = out_backprop.get("range")
        ori_data_format = out_backprop.get("ori_format")
        ori_paras = {
            "input_size": input_size, "x": out_backprop, "filters": filter, "bias": None, "offset_w": None,
            "y": input_grad,
            "strides": strides, "pads": pads, "dilations": dilations, "data_format": data_format,
            "output_padding": (0, 0, 0, 0), "offset_x": 0, "kernel_name": kernel_name
        }
        conv2d_tranpose = Conv2dTransposeParaProcess(ori_paras)
        conv2d_tranpose.get_attr_nchw(data_format)
        dy_shape_nchw = conv2d_tranpose.get_input_nchw(out_backprop.get("ori_shape"), out_backprop.get("ori_format"))
        filter_shape_nchw = conv2d_tranpose.get_input_nchw(filter.get("ori_shape"), filter.get("ori_format"))
        _, dy_range_nchw = conv2d_tranpose.get_input_nchw(dy_shape_nchw, ori_data_format, dy_range)
        if is_single_point:
            dy_range_nchw[3] = [dy_w_range_max, dy_w_range_max]
        else:
            dy_range_nchw[3] = [dy_range_nchw[3][0], min(dy_w_range_max, dy_range_nchw[3][1])]
        if out_backprop["ori_shape"][out_backprop.get("ori_format").find("W")] > dy_range_nchw[3][1]:
            err_man.raise_err_specific_user("depthwise_conv2d_backprop_input",
                                            "invalid out_backprop ori_shape {}, w should not larger than {}".format(
                                                str(out_backprop.get("shape")), dy_range_nchw[3][1]))
        dx_range_nchw, _, new_dy_range = conv2d_tranpose.get_input_range(filter_shape_nchw, dy_range_nchw)
        input_grad["range"] = [dx_range_nchw[0], [input_grad["shape"][1], input_grad["shape"][1]], dx_range_nchw[2],
                               dx_range_nchw[3],
                               [input_grad["shape"][4], input_grad["shape"][4]]]
        out_backprop["range"] = list(out_backprop["range"])
        out_backprop["ori_range"] = list(out_backprop["ori_range"])
        out_backprop["range"][out_backprop.get("format").find("H") - 1] = new_dy_range[2]
        out_backprop["ori_range"][out_backprop.get("ori_format").find("H")] = new_dy_range[2]
        out_backprop["range"][out_backprop.get("format").find("W") - 1] = new_dy_range[3]
        out_backprop["ori_range"][out_backprop.get("ori_format").find("W")] = new_dy_range[3]
        for name, tensor in have_range.items():
            # modify tesnors have range
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
            tensor["shape"] = [-1, tensor["shape"][1], -1, -1, tensor["shape"][4]]

        result.append([input_size, filter, out_backprop, input_grad, strides, dilations, pads,
                       data_format, kernel_name])
    return result


def _collect_ori_tensors(ori_paras):
    """
    get valid tensors
    """
    ori_tensors = {}
    for key, value in ori_paras.items():
        valid_tensor = isinstance(value, dict) \
                       and isinstance(value.get("ori_shape"), (list, tuple)) \
                       and len(value.get("ori_shape")) > 0
        if valid_tensor:
            ori_tensors[key] = value
    return ori_tensors


def _depthwise_conv2d_backprop_input_compute(
        input_size, filters, out_backprop, input_grad, strides, pads, dilations=(1, 1, 1, 1), data_format='NHWC',
        kernel_name='depthwise_conv2d_backprop_input'):  # pylint: disable=invalid-name, R0913
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
                   "res_dtype": input_grad.get("dtype"),
                   "kernel_name": kernel_name,
                   "group_dict": paras.get("group_para"),
                   "correct_range_flag": paras.get("correct_range_flag", False),
                   "ori_tensors": _collect_ori_tensors(ori_paras),
                   "op_type": "depthwise_conv2d_backprop_input"})

    return {'op_placeholder': [paras.get("input_tensor"), paras.get("filter_tensor"), paras.get("dy_tensor")],
            'op_res': [dedx]}


@register_operator('DepthwiseConv2DBackpropInput')
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
