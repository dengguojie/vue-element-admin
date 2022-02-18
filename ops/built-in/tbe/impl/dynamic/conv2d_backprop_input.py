#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import
import warnings
from impl.util import util_deconv_comm
from impl.util import util_select_op_base
from impl.util.util_conv2d import transform_shape_with_format
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import ceil_div
from impl.util.util_cube_dynamic import Conv2dBackpropParaProcess
from impl.util.util_cube_dynamic import Conv2dTransposeParaProcess
from impl.util.util_cube_dynamic import gen_conv_shape_range
from impl.util.util_cube_dynamic import modify_w_range_max
from impl.util.util_cube_dynamic import modify_dy_w_range_max_opti
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_cube_dynamic import check_graph_mode
from impl.util.util_cube_dynamic import check_para_fuzz_compile
from impl.util.util_cube_dynamic import set_default_para

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3
ORI_SHAPE_LEN = 4
SHAPE_LEN = 5
L1FUSION_INPUT_CTR = 2
OP_TYPE = "conv2d_backprop_input"
DATA_FORMAT_WHITE_LIST = ["NCHW", "NHWC"]
TAR_FORMAT = "NCHW"
MAX_N_FUZZ_BUILD = 2**31 - 1
MAX_HW_FUZZ_BUILD = 4096
LOWER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [2], "type": ["lower_limit"]}}]
UPPER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [2], "type": ["upper_limit"]}}]


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
                                         generalize_config=None):
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

    generalize_config: generalization mode, string.

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    support_mode = ["keep_rank"]
    is_generalize_config = (generalize_config is not None and generalize_config.get("mode") in support_mode)
    if not is_generalize_config:
        return
    result = []
    is_graph_mode = check_graph_mode(out_backprop)
    check_result = check_para_fuzz_compile(out_backprop, y, dilations, is_graph_mode, OP_TYPE)
    if check_result:
        return check_result
    out_backprop = gen_conv_shape_range(out_backprop, OP_TYPE, is_graph_mode)
    is_pass_check, dedy_modify = modify_dy_w_range_max_opti(out_backprop, filter, strides, data_format,
                                                            OP_TYPE, is_graph_mode)
    if not is_pass_check:
        return dedy_modify
    out_backprop = dedy_modify
    # if over l1 size then modify w range
    upper_range_result = modify_w_range_max(y, filter, out_backprop, strides, data_format, OP_TYPE, is_graph_mode)
    dy_h_range_max = upper_range_result.get("dedy_h_max")
    dy_w_range_max = upper_range_result.get("w_max")
    is_single_point = upper_range_result.get("is_single_point")
    graph_l1_invalid = is_graph_mode and upper_range_result.get("is_exceed_l1")
    single_l1_invalid = not is_graph_mode and upper_range_result.get("is_exceed_l1")
    a = ''
    if graph_l1_invalid:
        a = UPPER_STR
    if single_l1_invalid:
        a = LOWER_STR
    if a:
        return a
    if not is_graph_mode:
        # get dx_range depends on dy_range
        ori_data_format = out_backprop.get("ori_format")
        out_backprop_shape = out_backprop.get("ori_shape")
        y_data_format = y.get("ori_format")
        y_shape = y.get("ori_shape")
        y_shape_h = y_shape[y_data_format.find("H")]
        y_shape_w = y_shape[y_data_format.find("W")]
        out_backprop_shape_h = out_backprop_shape[ori_data_format.find("H")]
        out_backprop_shape_w = out_backprop_shape[ori_data_format.find("W")]
        pads_new = pads
        set_pads = out_backprop_shape_h == ceil_div(y_shape_h, strides[data_format.find("H")]) and \
                   out_backprop_shape_w == ceil_div(y_shape_w, strides[data_format.find("W")])
        if set_pads:
            pads_new = [-1, -1, -1, -1]
        ori_paras = {
            "input_size": input_size, "x": out_backprop, "filters": filter, "bias": None, "offset_w": None, "y": y,
            "strides": strides, "pads": pads_new, "dilations": dilations, "groups": groups, "data_format": data_format,
            "output_padding": (0, 0, 0, 0), "offset_x": 0, "kernel_name": kernel_name
        }
        conv2d_tranpose = Conv2dTransposeParaProcess(ori_paras)
        conv2d_tranpose.get_attr_nchw(data_format)
        dy_shape_nchw = conv2d_tranpose.get_input_nchw(out_backprop.get("ori_shape"), out_backprop.get("ori_format"))
        filter_shape_nchw = conv2d_tranpose.get_input_nchw(filter.get("ori_shape"), filter.get("ori_format"))
        _, dy_range_nchw = conv2d_tranpose.get_input_nchw(dy_shape_nchw, ori_data_format, out_backprop.get("ori_range"))
        dy_range_nchw[2] = [dy_range_nchw[2][0], min(dy_h_range_max, dy_range_nchw[2][1])]
        dy_range_nchw[3] = [dy_range_nchw[3][0], min(dy_w_range_max, dy_range_nchw[3][1])]
        if is_single_point:
            dy_range_nchw[3] = [dy_w_range_max, dy_w_range_max]
        if out_backprop["ori_shape"][out_backprop.get("ori_format").find("W")] > dy_range_nchw[3][1]:
            warnings.warn(OP_TYPE, "{}, invalid out_backprop ori_shape {}, w should not larger than {}".format(
                                    OP_TYPE, str(out_backprop.get("shape")), dy_range_nchw[3][1]))
            return LOWER_STR
        dx_range_nchw, _, new_dy_range = conv2d_tranpose.get_input_range(filter_shape_nchw, dy_range_nchw)
        dx_range_nchw[1] = [filter_shape_nchw[1] * groups, filter_shape_nchw[1] * groups]
        out_backprop["ori_range"] = list(out_backprop["ori_range"])
        out_backprop["ori_range"][out_backprop.get("ori_format").find("H")] = new_dy_range[2]
        out_backprop["ori_range"][out_backprop.get("ori_format").find("W")] = new_dy_range[3]
        have_range = {"inputs": out_backprop, "outputs": y}
        for _, tensor in have_range.items():
            # modify tesnors have range
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == TAR_FORMAT else [-1, -1, -1, tensor["ori_shape"][3]]

        input_size["const_value"] = None
        input_size["const_value_range"] = transform_shape_with_format(TAR_FORMAT, data_format,
                                                                      dx_range_nchw, DATA_FORMAT_WHITE_LIST)
    result.append([input_size, filter, out_backprop, y, {"strides": strides}, {"pads": pads},
                   {"dilations": dilations}, {"groups": groups}, {"data_format": data_format}])
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
@para_check.check_input_type((dict, tvm.tensor.Tensor), (dict, tvm.tensor.Tensor), (dict, tvm.tensor.Tensor),
                             dict, (tuple, list), (tuple, list), (tuple, list), int, str, str)
def conv2dbp_input_fusion_compute(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                  filters, out_backprop, y, strides, pads, dilations=(1, 1, 1, 1),
                                  groups=1, data_format='NHWC', kernel_name='conv2d_backprop_input'):
    """
    algorithm: conv2d_backprop_input

    Parameters
    ----------
    input_size: Tensor or dict, will not be used input tensor size.

    filter: Tensor or dict w, convolution kernel.

    out_backprop: Tensor or dict, gradients.

    y: dict with keys(ori_shape, ori_format, dtype and range) conv2d_backprop_input output tensor

    strides: tuple/list of 4 integers, filter move stride

    pads: tuple/list of 4 integers, [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers, filter expand size of dilated conv2d_backprop_input

    groups: int, param for group conv2d_backprop_input

    data_format: str, An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
    Specify the data format of the input and output data.

    kernel_name: str, kernel name, default value is "conv2d_backprop_input"

    Returns
    -------
    None
    """

    # set fusion build config
    build_cfg = {"constant_realize_extent_in_infer_bound": False}
    tbe_register.set_fusion_buildcfg("Conv2DBackpropInput", build_cfg)

    res = _conv2d_backprop_input_compute(input_size, filters, out_backprop, y, strides,
                                          pads, dilations, groups, data_format, kernel_name)
    if isinstance(out_backprop, tvm.tensor.Tensor):
        return res.get('op_res')[0]
    return res


def _conv2d_backprop_input_compute(input_size, filters, out_backprop, y, strides, pads,
                                   dilations=(1, 1, 1, 1), groups=1, data_format='NHWC',
                                   kernel_name='conv2d_backprop_input'):  # pylint: disable=invalid-name, R0913
    ori_paras = {
        "input_size": input_size, "filters": filters, "out_backprop": out_backprop, "y": y,
        "strides": strides, "pads": pads, "dilations": dilations, "groups": groups, "data_format": data_format,
        "kernel_name": kernel_name
    }

    default_para = set_default_para()
    if isinstance(input_size, dict) and not input_size.get("ori_shape"):
        ori_paras["input_size"]["ori_shape"] = default_para["input_size"]["ori_shape"]
    conv2dbp_para = Conv2dBackpropParaProcess(ori_paras)
    conv2dbp_para.config_paras()
    res_dtype = y.get("dtype").lower()
    attrs_info = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "groups": groups,
        "data_format": data_format
    }
    dedx = tbe.conv2d_backprop_input(filters=conv2dbp_para.tensors.get("filter_tensor"),
                                     out_backprop=conv2dbp_para.tensors.get("dy_tensor"),
                                     filter_sizes=conv2dbp_para.shape.get("filter_shape_nchw"),
                                     input_sizes=conv2dbp_para.shape.get("dx_shape_nchw"),
                                     para_dict={
                                         "strides":
                                             (conv2dbp_para.strides[H_DIM], conv2dbp_para.strides[W_DIM]),
                                         "padding": conv2dbp_para.pads,
                                         "dilations": conv2dbp_para.dilations,
                                         "res_dtype": res_dtype,
                                         "kernel_name": kernel_name,
                                         "group_dict": conv2dbp_para.attrs.get("group_para"),
                                         "correct_range_flag": conv2dbp_para.attrs.get("correct_range_flag", False),
                                         "binary_mode": conv2dbp_para.binary_mode,
                                         "ori_tensors": _collect_ori_tensors(ori_paras),
                                         "op_type": "Conv2DBackpropInput"
                                     })

    return {'op_placeholder': [conv2dbp_para.tensors.get("input_tensor"),
                               conv2dbp_para.tensors.get("filter_tensor"),
                               conv2dbp_para.tensors.get("dy_tensor")],
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
