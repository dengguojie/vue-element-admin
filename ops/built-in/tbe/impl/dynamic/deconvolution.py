#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

deconvolution
"""

from __future__ import absolute_import

from impl.util import util_deconv_comm
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import DeconvolutionParaProcess
from impl.util.util_cube_dynamic import check_graph_mode
from impl.util.util_cube_dynamic import set_default_para
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import error_manager_cube
from impl.util.util_cube_dynamic import gen_conv_shape_range
from impl.util.util_cube_dynamic import modify_w_range_max
from impl.util.util_cube_dynamic import modify_dy_w_range_max_opti

H_DIM = 2
W_DIM = 3
SHAPE_LEN = 5
ORI_SHAPE_LEN = 4
L1FUSION_INPUT_CTR = 2
OP_TYPE = "deconvolution"
LOWER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": ["lower_limit"]}}]


def get_op_support_info(x, filter, bias, offset_w, y, strides,
                        pads, dilations=(1, 1, 1, 1),
                        groups=1, data_format="NHWC", offset_x=0,
                        kernel_name="deconvolution"):
    """
    get the deconvolution split info

    """
    format_x = x.get("format")
    dtype_x = x.get("dtype")
    shape_x = x.get("ori_shape")
    shape_filters = util_deconv_comm.get_filter_shape(filter.get("ori_format"), filter.get("ori_shape"))
    if list(shape_x) != [-2]:
        shape_x = util_deconv_comm.get_filter_shape(x.get("ori_format"), shape_x)

    head_overlap_h = -1 if (shape_filters[2] == 1 and strides[0] == 1) else 0
    tail_overlap_h = head_overlap_h
    head_overlap_w = -1 if (shape_filters[3] == 1 and strides[1] == 1) else 0
    tail_overlap_w = head_overlap_w

    # input/output Serial， axis Serial, (headoverlap, tailoverlap, 0 means with overlap, -1 means without it)
    if format_x == "NC1HWC0":
        # cut N
        axis_split_matrix = [
            [util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
             util_select_op_base.SplitOutput([0, [0]])]
        ]
        # cut H
        if head_overlap_h == -1 or (list(shape_x) != [-2] and shape_x[2] > 0):
            axis_split_matrix += [
                [util_select_op_base.SplitInput([0, [2], [head_overlap_h], [tail_overlap_h]]),
                 util_select_op_base.SplitOutput([0, [2]])]
            ]
        # cut W
        if head_overlap_h == -1 or (list(shape_x) != [-2] and shape_x[3] > 0):
            axis_split_matrix += [
                [util_select_op_base.SplitInput([0, [3], [head_overlap_w], [tail_overlap_w]]),
                 util_select_op_base.SplitOutput([0, [3]])]
            ]
        # cut Cin
        c_axis = 0 if dtype_x == "float16" else 1
        head_overlap_c = 0 if dtype_x == "float16" else -1
        tail_overlap_c = head_overlap_c
        if bias:
            axis_split_matrix_bias = [
                [util_select_op_base.SplitInput([1, [c_axis], [head_overlap_c], [tail_overlap_c]],
                                                [2, [0], [-1], [-1]]),
                 util_select_op_base.SplitOutput([0, [1]])],
            ]
        else:
            axis_split_matrix_bias = [
                [util_select_op_base.SplitInput([1, [c_axis], [head_overlap_c], [tail_overlap_c]]),
                 util_select_op_base.SplitOutput([0, [1]])],
            ]
        axis_split_matrix += axis_split_matrix_bias
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, None)

    return op_cal_info_in_json


@tbe_register.register_param_generalization("Deconvolution")
def deconvolution_generalization(x, filter, bias, offset_w, y, strides, pads, dilations=(1, 1, 1, 1),
                                 groups=1, data_format='NHWC', offset_x=0, kernel_name=OP_TYPE,
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

    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    kernel_name: str
            kernel name, default value is "deconvolution"

    generalize_config: generalization mode, string.

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    support_mode = ["keep_rank"]
    if generalize_config["mode"] not in support_mode:
        error_manager_cube.raise_err_specific_user(OP_TYPE,
                                                   "invalid generalize mode {}, only support {}".format(
                                                       str(generalize_config["mode"]), str(support_mode)))
    result = []
    is_graph_mode = check_graph_mode(x)
    if generalize_config["mode"] == "keep_rank":  # fuzz build situation
        # unknow_rank x ori_shape is [-2], others' shape length is 4
        unknow_rank = len(x["ori_shape"]) == 1 and x["ori_shape"][0] == -2
        if unknow_rank:
            error_manager_cube.raise_err_specific_user(OP_TYPE, "not support unknow_rank under mode {}".format(
                generalize_config["mode"]))
        have_range_infor = {"x": x, "y": y}
        support_format = ["NCHW", "NHWC"]
        for name, tensor_infor in have_range_infor.items():
            if tensor_infor.get("ori_format") not in support_format:
                error_manager_cube.raise_err_specific_user(OP_TYPE,
                                                           "invalid {} ori_format {}, only support {}".format(
                                                               name, str(tensor_infor.get("ori_format")),
                                                               str(support_format)))
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor_infor.get("ori_shape"), (list, tuple)) and \
            len(tensor_infor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                error_manager_cube.raise_err_specific_user(OP_TYPE,
                                                           "invalid {} ori_shape {}, only support {}d".format(
                                                               name, str(tensor_infor.get("ori_shape")),
                                                               str(ORI_SHAPE_LEN)))
        # if over l1 size then modify w range
        strides_4d = [1, 1, strides[0], strides[1]]
        x = gen_conv_shape_range(x, OP_TYPE, is_graph_mode)
        is_pass_check, dedy_modify = modify_dy_w_range_max_opti(x, filter, strides_4d, data_format, OP_TYPE)
        if not is_pass_check:
            return dedy_modify
        x = dedy_modify
        upper_range_result = modify_w_range_max(y,
                                                filter,
                                                x,
                                                strides_4d,
                                                data_format,
                                                OP_TYPE)
        dy_h_range_max = upper_range_result.get("dedy_h_max")
        dy_w_range_max = upper_range_result.get("w_max")
        is_single_point = upper_range_result.get("is_single_point")
        if upper_range_result.get("is_exceed_l1"):
            return LOWER_STR

        # modify dy_range
        dy_range = x.get("ori_range")
        ori_data_format = x.get("ori_format")
        ori_paras = {
            "x": x, "filters": filter, "bias": None, "offset_w": None, "y": y,
            "strides": strides_4d, "pads": pads, "dilations": dilations, "data_format": data_format,
            "offset_x": 0, "kernel_name": kernel_name
        }
        deconvolution_para = DeconvolutionParaProcess(ori_paras)
        dy_shape_nchw = deconvolution_para.get_input_nchw(x.get("ori_shape"), x.get("ori_format"))
        filter_shape_nchw = deconvolution_para.get_input_nchw(filter.get("ori_shape"), filter.get("ori_format"))
        _, dy_range_nchw = deconvolution_para.get_input_nchw(dy_shape_nchw, ori_data_format, dy_range)
        dy_range_nchw[2] = [dy_range_nchw[2][0], min(dy_h_range_max, dy_range_nchw[2][1])]
        if is_single_point:
            dy_range_nchw[3] = [dy_w_range_max, dy_w_range_max]
        else:
            dy_range_nchw[3] = [dy_range_nchw[3][0], min(dy_w_range_max, dy_range_nchw[3][1])]
        if x["ori_shape"][x.get("ori_format").find("W")] > dy_range_nchw[3][1]:
            error_manager_cube.raise_err_specific_user(OP_TYPE,
                                            "invalid out_backprop ori_shape {}, w should not larger than {}".format(
                                                str(x.get("ori_shape")), dy_range_nchw[3][1]))
        _, _, new_dy_range = deconvolution_para.get_input_range(filter_shape_nchw, dy_range_nchw)
        x["ori_range"] = list(x["ori_range"])
        x["ori_range"][x.get("ori_format").find("H")] = new_dy_range[2]
        x["ori_range"][x.get("ori_format").find("W")] = new_dy_range[3]

        for name, tensor in have_range_infor.items():
            # modify tesnors have range
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
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
