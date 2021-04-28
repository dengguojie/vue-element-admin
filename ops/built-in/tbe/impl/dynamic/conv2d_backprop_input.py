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
from impl.util.util_cube_dynamic import Conv2dBackpropParaProcess
from impl.util.util_cube_dynamic import Conv2dTransposeParaProcess
from impl.util.util_cube_dynamic import set_default_para
from impl.util import fusion_util
from te.platform.fusion_manager import get_fusion_build_cfg
from impl.util.platform_adapter import register_operator_compute


NONETYPE = type(None)
H_DIM = 2
W_DIM = 3
ORI_SHAPE_LEN = 4
SHAPE_LEN = 5


@tbe_register.register_param_generalization("Conv2DBackpropInput")
def conv2d_backprop_input_generalization(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                         filter, out_backprop, y, strides,
                                         pads, dilations=(1, 1, 1, 1),
                                         groups=1, data_format="NHWC",
                                         kernel_name="conv2d_backprop_input",
                                         generalize_config={"mode": "keep_rank"}):
    """
    conv2d backprop input generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to conv2d_backprop_input

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    support_mode = ["keep_rank"]
    if generalize_config["mode"] not in support_mode:
        err_man.raise_err_specific_user("conv2d_backprop_input", "invalid generalize mode {}, only support {}".format(
            str(generalize_config["mode"]), str(support_mode)))
    result = []
    if generalize_config["mode"] == "keep_rank":  # fuzz build situation
        # unknow_rank inputs ori_shape is [-2], others' shape length is 4
        unknow_rank = len(out_backprop["ori_shape"]) == 1 and out_backprop["ori_shape"][0] == -2
        if unknow_rank:
            err_man.raise_err_specific_user("conv2d_backprop_input", "not support unknow_rank under mode {}".format(
                generalize_config["mode"]))
        have_range = {"inputs": out_backprop, "outputs": y}
        support_format = ["NCHW", "NHWC"]
        for name, tensor in have_range.items():
            # modify tesnors have range
            if tensor.get("ori_format") not in support_format:
                err_man.raise_err_specific_user("conv2d_backprop_input",
                                                "invalid {} ori_format {}, only support {}".format(
                                                    name, str(tensor.get("ori_format")), str(support_format)))
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                err_man.raise_err_specific_user("conv2d_backprop_input",
                                                "invalid {} ori_shape {}, only support {}d".format(
                                                    name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
            valid = isinstance(tensor.get("shape"), (list, tuple)) and len(tensor["shape"]) == SHAPE_LEN
            if not valid:
                err_man.raise_err_specific_user("conv2d_backprop_input",
                                                "invalid {} ori_shape {}, only support {}d".format(
                                                    name, str(tensor.get("shape")), str(SHAPE_LEN)))
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
            tensor["shape"] = [-1, tensor["shape"][1], -1, -1, tensor["shape"][4]]

        # get dx_range depends on dy_range
        dy_range = out_backprop.get("range")
        data_format = out_backprop.get("ori_format")
        ori_paras = {
            "input_size": input_size, "x": out_backprop, "filters": filter, "bias": None, "offset_w": None, "y": y,
            "strides": strides, "pads": pads, "dilations": dilations, "groups": groups, "data_format": data_format,
            "output_padding": (0, 0, 0, 0), "offset_x": 0, "kernel_name": kernel_name
        }
        conv2d_tranpose = Conv2dTransposeParaProcess(ori_paras)
        dy_shape_nchw = conv2d_tranpose.get_input_nchw(out_backprop.get("ori_shape"), out_backprop.get("ori_format"))
        filter_shape_nchw = conv2d_tranpose.get_input_nchw(filter.get("ori_shape"), filter.get("ori_format"))
        _, dy_range_nchw = conv2d_tranpose.get_input_nchw(dy_shape_nchw, data_format, dy_range)
        dx_range_nchw, _, _ = conv2d_tranpose.get_input_range(filter_shape_nchw, dy_range_nchw)
        y["range"] = [dx_range_nchw[0], [y["shape"][1], y["shape"][1]], dx_range_nchw[2], dx_range_nchw[3],
                      [y["shape"][4], y["shape"][4]]]

        result.append([input_size, filter, out_backprop, y, strides, pads, dilations,
                       groups, data_format, kernel_name])
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


@register_operator_compute("Conv2DBackpropInput", op_mode="dynamic", support_fusion=True)
def conv2dbp_input_fusion_compute(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                  filters, out_backprop, y, strides, pads, dilations=(1, 1, 1, 1),
                                  groups=1, data_format='NHWC', kernel_name='conv2d_backprop_input'):
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

    fusion_util.check_fusion_input([input_size, filters, out_backprop])
    # set fusion build config
    build_cfg = get_fusion_build_cfg()
    build_cfg['constant_realize_extent_in_infer_bound'] = False
    
    return _conv2d_backprop_input_compute(input_size, filters, out_backprop, y, strides,
                                          pads, dilations, groups, data_format, kernel_name)


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

    dedx = tbe.conv2d_backprop_input(filters=paras.get("filter_tensor"),
                                     out_backprop=paras.get("dy_tensor"),
                                     filter_sizes=paras.get("filter_shape"),
                                     input_sizes=paras.get("input_size"),
                                     para_dict={
                                         "strides":
                                         (conv2dbp_para.strides[H_DIM], conv2dbp_para.strides[W_DIM]),
                                         "padding": conv2dbp_para.pads,
                                         "dilations": conv2dbp_para.dilations,
                                         "res_dtype": default_para.get("res_dtype"),
                                         "kernel_name": kernel_name,
                                         "group_dict": paras.get("group_para"),
                                         "correct_range_flag": paras.get("correct_range_flag", False),
                                         "ori_tensors": _collect_ori_tensors(ori_paras),
                                         "op_type": "Conv2DBackpropInput"
                                     })

    return {'op_placeholder': [paras.get("input_tensor"), paras.get("filter_tensor"), paras.get("dy_tensor")],
            'op_res': [dedx]}


@operation.register_operator('Conv2DBackpropInput')
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

    with tbe.compute():
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
