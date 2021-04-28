#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

deconvolution
"""

from __future__ import absolute_import

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import DeconvolutionParaProcess
from impl.util.util_cube_dynamic import set_default_para
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import error_manager_cube

H_DIM = 2
W_DIM = 3
SHAPE_LEN = 5
ORI_SHAPE_LEN = 4


@tbe_register.register_param_generalization("Deconvolution")
def deconvolution_generalization(x, filter, bias, offset_w, y, strides, pads, dilations=(1, 1, 1, 1),
                                 groups=1, data_format='NHWC', offset_x=0, kernel_name="deconvolution",
                                 generalize_config={"mode": "keep_rank"}):
    """
    deconvolution generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to deconvolution

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    support_mode = ["keep_rank"]
    if generalize_config["mode"] not in support_mode:
        error_manager_cube.raise_err_specific_user("deconvolution",
                                                   "invalid generalize mode {}, only support {}".format(
                                                       str(generalize_config["mode"]), str(support_mode)))
    result = []
    if generalize_config["mode"] == "keep_rank":  # fuzz build situation
        # unknow_rank x ori_shape is [-2], others' shape length is 4
        unknow_rank = len(x["ori_shape"]) == 1 and x["ori_shape"][0] == -2
        if unknow_rank:
            error_manager_cube.raise_err_specific_user("deconvolution", "not support unknow_rank under mode {}".format(
                generalize_config["mode"]))
        have_range = {"x": x, "y": y}
        support_format = ["NCHW", "NHWC"]
        for name, tensor in have_range.items():
            # modify tesnors have range
            if tensor.get("ori_format") not in support_format:
                error_manager_cube.raise_err_specific_user("deconvolution",
                                                           "invalid {} ori_format {}, only support {}".format(
                                                               name, str(tensor.get("ori_format")),
                                                               str(support_format)))
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                error_manager_cube.raise_err_specific_user("deconvolution",
                                                           "invalid {} ori_shape {}, only support {}d".format(
                                                               name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
            valid = isinstance(tensor.get("shape"), (list, tuple)) and len(tensor["shape"]) == SHAPE_LEN
            if not valid:
                error_manager_cube.raise_err_specific_user("deconvolution",
                                                           "invalid {} ori_shape {}, only support {}d".format(
                                                               name, str(tensor.get("shape")), str(SHAPE_LEN)))
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
            tensor["shape"] = [-1, tensor["shape"][1], -1, -1, tensor["shape"][4]]
        result.append([x, filter, bias, offset_w, y, strides, pads, dilations,
                       groups, data_format, offset_x, kernel_name])
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


def _deconvolution_compute(x, filter, bias, offset_w,
                           y, strides, pads,
                           dilations=(1, 1, 1, 1),
                           groups=1, data_format='NHWC', offset_x=0,
                           kernel_name='deconvolution'):
    ori_paras = {
        "x": x, "filters": filter, "bias": bias, "offset_w": offset_w, "y": y,
        "strides": strides, "pads": pads, "dilations": dilations, "groups": groups,
        "data_format": data_format, "offset_x": offset_x, "kernel_name": kernel_name
    }

    conv2dbp_para = DeconvolutionParaProcess(ori_paras)
    paras = conv2dbp_para.config_paras()
    default_para = set_default_para()
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
                                         "group_dict": paras.get("group_para"),
                                         "correct_range_flag": paras.get("correct_range_flag", False),
                                         "ori_tensors": _collect_ori_tensors(ori_paras),
                                         "op_type": "Deconvolution"
                                     })
    if bias:
        return {'op_placeholder': [paras.get("x_tensor"), paras.get("filter_tensor"), paras.get("bias_tensor")],
                'op_res': [dedx]}
    return {'op_placeholder': [paras.get("x_tensor"), paras.get("filter_tensor")], 'op_res': [dedx]}


@register_operator('Deconvolution')
@para_check.check_input_type(dict, dict, (type(None), dict), (type(None), dict), dict, (tuple, list),
                             (tuple, list), (tuple, list), int, str, int, str,
                             (type(None), dict))
def deconvolution(x, filter, bias, offset_w, y, strides,
                  pads, dilations=(1, 1, 1, 1),
                  groups=1, data_format="NHWC", offset_x=0,
                  kernel_name="deconvolution"):
    """
    algorithm: deconvolution

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, dtype)
        The shape of gradients.

    filter: dict with keys(ori_shape, ori_format, dtype)
            convolution kernel

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
        Input offset_w tensor.

    y: dict with keys(ori_shape, ori_format, dtype and range)
       deconvolution output tensor

    strides: tuple/list of 2 integers
             filter move stride

    pads: tuple/list of 4 integers
          str: "SAME" or "VALID"
          tuple/list of 4 integers: [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated deconvolution
    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "deconvolution"

    Returns
    -------
    None
    """

    with tbe.compute():
        res = _deconvolution_compute(
            x, filter, bias, offset_w, y,
            strides, pads, dilations, groups, data_format, offset_x, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get('op_res'))

    tensor_list = res.get('op_placeholder') + res.get('op_res')
    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensor_list,
              'build_args': {'constant_realize_extent_in_infer_bound': False}}
    tbe.build(sch, config)
