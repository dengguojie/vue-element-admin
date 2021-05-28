#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import

from impl.util import fusion_util
from impl.util import util_deconv_comm
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_cube as err_man
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import Conv2dBackpropParaProcess
from impl.util.util_cube_dynamic import Conv2dTransposeParaProcess
from impl.util.util_cube_dynamic import set_default_para
from impl.util.util_cube_dynamic import modify_w_range_max
from impl.util.platform_adapter import register_operator_compute

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3
ORI_SHAPE_LEN = 4
SHAPE_LEN = 5
L1FUSION_INPUT_CTR = 2


def get_op_support_info(input_size, filter, out_backprop, y, strides,
                        pads, dilations=(1, 1, 1, 1), groups=1,
                        data_format="NHWC", kernel_name="conv2d_backprop_input"):
    """
    get the conv2d_backprop_input split info

    """
    h_pos = data_format.find("H")
    w_pos = data_format.find("W")
    shape_out_backprop = out_backprop.get("ori_shape")
    shape_filters = util_deconv_comm.get_filter_shape(filter.get("ori_format"),
                                                      filter.get("ori_shape"))
    if list(shape_out_backprop) != [-2]:
        shape_out_backprop = util_deconv_comm.get_filter_shape(out_backprop.get("ori_format"),
                                                               shape_out_backprop)

    head_overlap_h = -1 if (shape_filters[2] == 1 and strides[h_pos] == 1) else 0
    tail_overlap_h = head_overlap_h
    head_overlap_w = -1 if (shape_filters[3] == 1 and strides[w_pos] == 1) else 0
    tail_overlap_w = head_overlap_w

    format_out_backprop = out_backprop.get("format")
    # input/output Serial， axis Serial, (headoverlap, tailoverlap, 0 means with overlap, -1 means without it)
    if format_out_backprop == "NC1HWC0":
        # cut N
        axis_split_matrix = [
            [util_select_op_base.SplitInput([1, [0], [-1], [-1]]),
             util_select_op_base.SplitOutput([0, [0]])]
        ]
        # cut Cin
        axis_split_matrix += [
            [util_select_op_base.SplitInput([0, [0], [0], [0]]),
             util_select_op_base.SplitOutput([0, [1]])]
        ]
        # cut H
        if head_overlap_h == -1 or (list(shape_out_backprop) != [-2] and shape_out_backprop[2] > 0):
            axis_split_matrix += [
                [util_select_op_base.SplitInput([1, [2], [head_overlap_h], [tail_overlap_h]]),
                 util_select_op_base.SplitOutput([0, [2]])]
            ]
        # cut w
        if head_overlap_w == -1 or (list(shape_out_backprop) != [-2] and shape_out_backprop[3] > 0):
            axis_split_matrix += [
                [util_select_op_base.SplitInput([1, [3], [head_overlap_w], [tail_overlap_w]]),
                 util_select_op_base.SplitOutput([0, [3]])]
            ]
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, None
    )

    return op_cal_info_in_json


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
        # if over l1 size then modify w range
        dy_w_range_max, is_single_point = modify_w_range_max(y.get("ori_shape")[y.get("ori_format").find("W")],
                                                             filter.get("ori_shape")[
                                                                 filter.get("ori_format").find("W")],
                                                             filter.get("ori_shape")[
                                                                 filter.get("ori_format").find("H")],
                                                             out_backprop.get("ori_shape")[
                                                                 out_backprop.get("ori_format").find("W")],
                                                             strides[data_format.find("W")],
                                                             out_backprop.get("dtype").lower(),
                                                             filter.get("dtype").lower(),
                                                             "conv2d_backprop_input")

        # get dx_range depends on dy_range
        dy_range = out_backprop.get("range")
        ori_data_format = out_backprop.get("ori_format")
        ori_paras = {
            "input_size": input_size, "x": out_backprop, "filters": filter, "bias": None, "offset_w": None, "y": y,
            "strides": strides, "pads": pads, "dilations": dilations, "groups": groups, "data_format": data_format,
            "output_padding": (0, 0, 0, 0), "offset_x": 0, "kernel_name": kernel_name
        }
        conv2d_tranpose = Conv2dTransposeParaProcess(ori_paras)
        dy_shape_nchw = conv2d_tranpose.get_input_nchw(out_backprop.get("ori_shape"), out_backprop.get("ori_format"))
        filter_shape_nchw = conv2d_tranpose.get_input_nchw(filter.get("ori_shape"), filter.get("ori_format"))
        _, dy_range_nchw = conv2d_tranpose.get_input_nchw(dy_shape_nchw, ori_data_format, dy_range)
        if is_single_point:
            dy_range_nchw[3] = [dy_w_range_max, dy_w_range_max]
        else:
            dy_range_nchw[3] = [dy_range_nchw[3][0], min(dy_w_range_max, dy_range_nchw[3][1])]
        if out_backprop["ori_shape"][out_backprop.get("ori_format").find("W")] > dy_range_nchw[3][1]:
            err_man.raise_err_specific_user("conv2d_backprop_input",
                                            "invalid out_backprop ori_shape {}, w should not larger than {}".format(
                                                str(out_backprop.get("shape")), dy_range_nchw[3][1]))
        dx_range_nchw, _, _ = conv2d_tranpose.get_input_range(filter_shape_nchw, dy_range_nchw)
        y["range"] = [dx_range_nchw[0], [y["shape"][1], y["shape"][1]], dx_range_nchw[2], dx_range_nchw[3],
                      [y["shape"][4], y["shape"][4]]]
        out_backprop["range"] = list(out_backprop["range"])
        out_backprop["ori_range"] = list(out_backprop["ori_range"])
        out_backprop["range"][out_backprop.get("format").find("W") - 1] = dy_range_nchw[3]
        out_backprop["ori_range"][out_backprop.get("ori_format").find("W")] = dy_range_nchw[3]
        for name, tensor in have_range.items():
            # modify tesnors have range
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
            tensor["shape"] = [-1, tensor["shape"][1], -1, -1, tensor["shape"][4]]

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
    build_cfg = tbe_register.get_fusion_buildcfg()
    build_cfg['constant_realize_extent_in_infer_bound'] = False

    return _conv2d_backprop_input_compute(input_size, filters, out_backprop, y, strides,
                                          pads, dilations, groups, data_format, kernel_name)


def _conv2d_backprop_input_compute(input_size, filters, out_backprop, y, strides, pads,
                                   dilations=(1, 1, 1, 1), groups=1, data_format='NHWC',
                                   kernel_name='conv2d_backprop_input'):  # pylint: disable=invalid-name, R0913
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


@tbe_register.register_operator('Conv2DBackpropInput')
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
