# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
conv2d_backprop_input_d
"""
from impl.util import util_deconv_comm
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

# the dim of shape in conv2d_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4

# each axis of shape must less than 1000000
DEFAULT_MAX_SHAPE_NUM = 1000000

# 2 means L1 enable
L1FUSION_INPUT_CTR = 2

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255
# stride must be >= 1
STRIDE_HW_MIN = 1

# pad must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255
# filterH, filterW must be >= 1
FILTER_HW_MIN = 1

INOUT_HW_MIN = 1
INOUT_H_MAX = 200000
INOUT_W_MAX = 4096
CONV1D_W_MAX = 2147483647

# position index
N_DIM = 0
H_DIM = 2
W_DIM = 3


def _cal_min_l1space(out_backprop,  # pylint: disable=invalid-name
                     weight, y, strides, dilations, pads):
    """
    cal the mini l1space using in lxfusion

    Parameters
    ----------
    out_backprop: dict with keys(shape and dtype) or Tensor
        The shape of gradients.

    weight: dict with keys(shape and dtype) or Tensor
        input weight tensor

    y: dict with keys(shape and dtype)
        conv2d_backprop_input output tensor, dtype must be assigned.

    strides: list of 2 integers

    dilations: tuple/list of 2 integers

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].


    Returns
    -------
    res: the min l1 space of lxfusion
    """
    def _cal_al1_size():
        w_value = shape_out_backprop[3] * strides[1]
        if shape_res[3] > c0_size:
            h_value_max = filter_h_dilation + 1
        elif c0_size % shape_res[3] == 0:
            h_value_max = filter_h_dilation + c0_size // shape_res[3] - 1
        else:
            h_value_max = filter_h_dilation + c0_size // shape_res[3] + 1
        a_l1_size = h_value_max * w_value * c0_size_k * util_deconv_comm.BIT_RATIO_DICT.get(filters_dtype)
        return a_l1_size

    shape_filters = util_deconv_comm.get_filter_shape(weight.get("ori_format"), weight.get("ori_shape"))
    shape_out_backprop = util_deconv_comm.get_shape_out_backprop(
        out_backprop.get("ori_format"), out_backprop.get("ori_shape"))
    shape_res = util_deconv_comm.get_shape_res(y.get("ori_format"), y.get("ori_shape"))
    filters_dtype = weight.get("dtype")

    c0_size = tbe_platform.C0_SIZE
    c0_size_k = tbe_platform.CUBE_MKN[filters_dtype]['mac'][1]
    filter_h_dilation = (shape_filters[2] - 1) * dilations[0] + 1

    bl1_size = shape_filters[2] * shape_filters[3] * c0_size * c0_size_k * \
               util_deconv_comm.BIT_RATIO_DICT.get(filters_dtype)
    al1_size = 0
    if (list(pads) != [0, 0, 0, 0] or list(shape_filters[2:]) != [1, 1]) and list(strides) != [1, 1]:
        al1_size = _cal_al1_size()
    return al1_size + bl1_size


def get_op_support_info( # pylint: disable=invalid-name,R0913,R0914,W0613
        filter,
        out_backprop,
        y,
        input_size,
        strides,
        pads,
        dilations=(1, 1, 1, 1),
        groups=1,
        data_format="NHWC",
        kernel_name="conv2d_backprop_input",
):
    """
    get the deconvolution split

    Parameters
    ----------
    filter: dict with keys(shape and dtype) or Tensor
        input weight tensor

    out_backprop: dict with keys(shape and dtype) or Tensor
        The shape of gradients.

    y: dict with keys(shape and dtype)
        conv2d_backprop_input output tensor, dtype must be assigned.

    input_size: The shape of feature map.
        4-D with shape [batch, channels, height, weight].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_input.

    groups: int
        param for group conv2d_backprop_input. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_input".

    Returns
    -------
    res: the json of op info with split
    """
    h_pos = data_format.find("H")
    w_pos = data_format.find("W")
    shape_filters = util_deconv_comm.get_filter_shape(filter.get("ori_format"), filter.get("ori_shape"))
    head_overlap_h = -1 if (shape_filters[2] == 1 and strides[h_pos] == 1) else 0
    tail_overlap_h = head_overlap_h
    head_overlap_w = -1 if (shape_filters[3] == 1 and strides[w_pos] == 1) else 0
    tail_overlap_w = head_overlap_w

    format_out_backprop = out_backprop.get("format")
    # input/output Serial， axis Serial,  (headoverlap, tailoverlap, 0 means with overlap, -1 means without it)
    if format_out_backprop == "NC1HWC0":
        axis_split_matrix = [
            # cut N
            [ util_select_op_base.SplitInput([1, [0], [-1], [-1]]),
              util_select_op_base.SplitOutput([0, [0]])],
            # cut H
            [util_select_op_base.SplitInput([1, [2], [head_overlap_h], [tail_overlap_h]]),
             util_select_op_base.SplitOutput([0, [2]])],
            # cut W
            [util_select_op_base.SplitInput([1, [3], [head_overlap_w], [tail_overlap_w]]),
             util_select_op_base.SplitOutput([0, [3]])],
        ]
        # cut Cin
        axis_split_matrix += [
            [util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
             util_select_op_base.SplitOutput([0, [1]])],
        ]
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    stride_list = [strides[h_pos], strides[w_pos]]
    dilation_list = [dilations[h_pos], dilations[w_pos]]
    min_l1space = _cal_min_l1space(out_backprop, filter, y, stride_list, dilation_list, pads)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list,
                                                              L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


def _check_conv2dbp_input_para(  # pylint: disable=W0622,C0103,R0913,R0914
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    dilations,
    data_format,
    topi_flag=False,
):
    """
    check the inputpara and get input shape

    Parameters
    ----------
    filter: dict with keys(shape and dtype) or Tensor
        input weight tensor

    out_backprop: dict with keys(shape and dtype) or Tensor
        The shape of gradients.

    y: dict with keys(shape and dtype)
        conv2d_backprop_input output tensor, dtype must be assigned.

    input_size: The shape of feature map.
        4-D with shape [batch, channels, height, weight].

    strides: tuple/list of 4 integers
        filter move stride.

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_input.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    topi_flag: input para from topi or compute
        False: compute; True: topi. Default to False.

    Returns
    -------
    res: the shape, strides, dilations
    """

    if topi_flag:
        ori_shape_filters = filter.get("ori_shape")
        ori_shape_out_backprop = out_backprop.get("ori_shape")
        ori_shape_res = y.get("ori_shape")

        ori_format_filters = filter.get("ori_format")
        ori_format_out_backprop = out_backprop.get("ori_format")
        ori_format_res = y.get("ori_format")
    else:
        ori_shape_filters = [i.value for i in filter.op.attrs["ori_shape"]]
        ori_shape_out_backprop = [i.value for i in out_backprop.op.attrs["ori_shape"]]
        ori_shape_res = [i for i in y["ori_shape"]]

        ori_format_filters = filter.op.attrs["ori_format"]
        ori_format_out_backprop = out_backprop.op.attrs["ori_format"]
        ori_format_res = y["ori_format"]

    if list(input_size) != list(ori_shape_res):
        dict_args = {}
        dict_args["errCode"] = "E65007"
        dict_args["param1"] = "input_size"
        dict_args["param2"] = "ori_shape of y"
        dict_args["actual_value"] = "{}, {}".format(input_size, ori_shape_res)
        raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
    if len(strides) == 4:
        h_index = data_format.find("H")
        w_index = data_format.find("W")
        strides = [strides[h_index], strides[w_index]]

    shape_filters = util_deconv_comm.get_filter_shape(
        ori_format_filters, ori_shape_filters
    )

    shape_out_backprop = util_deconv_comm.get_shape_out_backprop(
        ori_format_out_backprop, ori_shape_out_backprop
    )

    shape_res = util_deconv_comm.get_shape_res(ori_format_res, ori_shape_res)

    dilations = util_deconv_comm.get_shape_dilation(data_format, dilations)

    return [shape_filters, shape_out_backprop, shape_res, strides, dilations]


def _support_situation(  # pylint: disable=W0622,C0103,R0913,R0914
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    pads,
    dilations,
    groups,
    data_format,
    kernel_name):
    """
    check the op support situation
    """
    para_check.check_kernel_name(kernel_name)
    res = _check_conv2dbp_input_para(
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    dilations,
    data_format,
    topi_flag=True,
    )
    shape_filters, shape_out_backprop, shape_res, strides, dilations = res

    util_deconv_comm.check_equal_rule(shape_out_backprop[0], shape_res[0],
                                      "batch of out_backprop", "batch of y")
    util_deconv_comm.check_equal_rule(shape_out_backprop[1], shape_filters[0],
                                      "channel of out_backprop", "batch of filter")
    util_deconv_comm.check_equal_rule(shape_res[1], shape_filters[1] * groups,
                                      "channel of y", "channel of filter * groups")
    if groups == 0:
        args_dict = {
            "errCode": "E60108",
            "reason": "groups can not be 0",
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
    if shape_out_backprop[1] % groups != 0:
        args_dict = {
            "errCode": "E60108",
            "reason": "channel of out_backprop % groups must be 0",
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
    if shape_res[1] % groups != 0:
        args_dict = {
            "errCode": "E60108",
            "reason": "channel of y % groups must be 0",
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
    util_deconv_comm.check_attr_range("dilations's H", dilations[2], DILATION_MIN, DILATION_MAX)
    util_deconv_comm.check_attr_range("dilations's W", dilations[3], DILATION_MIN, DILATION_MAX)
    util_deconv_comm.check_attr_range("strides's H", strides[0], STRIDE_HW_MIN)
    util_deconv_comm.check_attr_range("strides's W", strides[1], STRIDE_HW_MIN)

    util_deconv_comm.check_attr_range("filter's H", shape_filters[2], FILTER_HW_MIN)
    util_deconv_comm.check_attr_range("filter's W", shape_filters[3], FILTER_HW_MIN)

    pads = util_deconv_comm.get_padlist(pads, shape_res, strides, shape_filters, dilations)
    pad_up, pad_down, pad_left, pad_right = pads
    filter_h, filter_w = shape_filters[2:]
    dedy_h, dedy_w = shape_out_backprop[2:]
    fmap_h, fmap_w = shape_res[2:]

    filter_h_dilation = (filter_h - 1) * dilations[2] + 1
    filter_w_dilation = (filter_w - 1) * dilations[3] + 1
    fmap_h_padding = fmap_h + pad_up + pad_down
    fmap_w_padding = fmap_w + pad_left + pad_right
    inout_w_max = INOUT_W_MAX
    if fmap_h_padding == 1 and filter_h_dilation == 1 and strides[0] == 1:
        inout_w_max = CONV1D_W_MAX

    util_deconv_comm.check_attr_range("out_backprop's H", shape_out_backprop[2],
                                      INOUT_HW_MIN, INOUT_H_MAX)
    util_deconv_comm.check_attr_range("out_backprop's W", shape_out_backprop[3],
                                      INOUT_HW_MIN, inout_w_max)
    util_deconv_comm.check_attr_range("y's H", shape_res[2], INOUT_HW_MIN, INOUT_H_MAX)
    util_deconv_comm.check_attr_range("y's W", shape_res[3], INOUT_HW_MIN, inout_w_max)
    util_deconv_comm.check_attr_range("out_backprop's H after expands", dedy_h * strides[0],
                                      INOUT_HW_MIN, INOUT_H_MAX)
    if filter_h == 1 and filter_w == 1:
        util_deconv_comm.check_attr_range("out_backprop's W after expands",
                                          dedy_w * strides[0] * strides[1],
                                          INOUT_HW_MIN, inout_w_max)
    else:
        util_deconv_comm.check_attr_range("out_backprop's W after expands", dedy_w * strides[1],
                                          INOUT_HW_MIN, inout_w_max)

    if ((fmap_h - filter_h_dilation + pad_up + pad_down) // strides[0] + 1) != dedy_h:
        args_dict = {
            "errCode": "E60024",
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
    if ((fmap_w - filter_w_dilation + pad_left + pad_right) // strides[1] + 1) != dedy_w:
        args_dict = {
            "errCode": "E60025",
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))


def check_supported(
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    kernel_name="conv2d_backprop_input",
):
    """
    check the op support situation:

    | Name             | Field    | Scope
    -------------------|----------|--------------
    | input_size       | H or W   | [1, 4096]
    -------------------|----------|--------------
    | Filter           | H or W   | [1, 255]
    -------------------|----------|--------------
    | out_backprop     | H or W   | [1, 4096]
    -------------------|----------|--------------
    | y(fmap)          | H or W   | [1, 4096]
    -------------------|----------|--------------
    | Stride           | H or W   | [1, 63]
    -------------------|----------|--------------
    | Dilation         | H or W   | [1, 255]

    batch_input_size == batch_out_backprop

    batch_filter == channel_out_backprop

    channel_filter == channel_input_size * groups

    out_backprop_height == (fmap_height + pad_top + pad_bottom -
                          (dilation_h * (filter_height - 1) + 1))
                           / stride_h + 1

    out_backprop_width == (fmap_width + pad_left + pad_right -
                         (dilation_w * (filter_width - 1) + 1))
                          / stride_w + 1
    """
    shape_backprop = out_backprop.get("ori_shape")
    dynamic_flag = any([i < 0 for i in shape_backprop])
    if dynamic_flag:
        return True, ""
    try:
        _support_situation(filter, out_backprop, y, input_size, strides, pads,
                           dilations, groups, data_format, kernel_name)
        return True, ""
    except RuntimeError as e:
        reason = e.args[1]
        return False, reason


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.KERNEL_NAME,
)
def conv2d_backprop_input_d(  # pylint: disable=W0622,C0103,R0913,R0914
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    kernel_name="conv2d_backprop_input",
):
    """
    algorithm: conv2d_backprop_input

    Parameters
    ----------
    filter: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input weight tensor.

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype)
        The shape of gradients.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype)
        conv2d_backprop_input output tensor, dtype must be assigned.

    input_size: tuple/list of 4 integers
        The shape of feature map. 4-D with shape [batch, height, width, channels]
        or [batch, channels, height, filter].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_input. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_input. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_input".

    Returns
    -------
    None
    """

    ori_shape_filters = filter.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = y.get("ori_shape")

    filters_dtype = filter.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y.get("dtype")
    filter_ori_format = filter.get("ori_format")
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(
        ori_shape_filters,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )
    para_check.check_shape_rule(
        ori_shape_out_backprop,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )
    para_check.check_shape_rule(
        input_size,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )
    para_check.check_shape_rule(
        ori_shape_res,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )
    para_check.check_shape_rule(
        dilations,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )

    res = _check_conv2dbp_input_para(
        filter,
        out_backprop,
        y,
        input_size,
        strides,
        dilations,
        data_format,
        topi_flag=True,
    )
    shape_filters, shape_out_backprop, shape_res, strides, dilations = res
    shape_out_backprop_5hd = out_backprop.get("shape")
    shape_out_backprop = list(shape_out_backprop)
    shape_out_backprop[N_DIM] = shape_out_backprop_5hd[N_DIM]
    shape_out_backprop[H_DIM] = shape_out_backprop_5hd[H_DIM]
    shape_out_backprop[W_DIM] = shape_out_backprop_5hd[W_DIM]

    group_dict = util_deconv_comm.calculate_group(
        shape_out_backprop,
        shape_res,
        shape_filters,
        groups,
        filters_dtype,
        filter_ori_format
    )
    _conv2d_backprop_input_cce(
        shape_filters,
        shape_out_backprop,
        shape_res,
        strides,
        pads,
        dilations,
        filters_dtype,
        out_backprop_dtype,
        res_dtype,
        kernel_name,
        group_dict
    )


@tbe_platform.fusion_manager.register("conv2d_backprop_input_d")
def conv2d_backprop_input_d_compute(  # pylint: disable=C0103,W0622,R0913,R0914
    filter,
    out_backprop,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",  # pylint: disable=W0613
    kernel_name="conv2d_backprop_input",
):
    """
    used for fusion
    Parameters
    ----------
    filter: Tensor
        input weight tensor.

    out_backprop: Tensor
        conv2d output gradients tenosr.

    y: dict with keys(shape and dtype)
        conv2d_backprop_input output tensor, dtype must be assigned.

    input_size: tuple/list of 4 integers
        The shape of feature map. 4-D with shape [batch, height, width, channels]
        or [batch, channels, height, filter].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_input. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_input. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_input".

    Returns
    -------
    Tensor of conv2d_backprop_input
    """

    filters_dtype = filter.dtype
    out_backprop_dtype = out_backprop.dtype
    res_dtype = y["dtype"]
    filter_ori_format = filter.op.attrs["ori_format"].value
    res = _check_conv2dbp_input_para(
        filter,
        out_backprop,
        y,
        input_size,
        strides,
        dilations,
        data_format,
        topi_flag=False,
    )
    shape_filters, shape_out_backprop, shape_res, strides, dilations = res
    shape_out_backprop_5hd = [i.value for i in out_backprop.shape]
    shape_out_backprop = list(shape_out_backprop)
    shape_out_backprop[N_DIM] = shape_out_backprop_5hd[N_DIM]
    shape_out_backprop[H_DIM] = shape_out_backprop_5hd[H_DIM]
    shape_out_backprop[W_DIM] = shape_out_backprop_5hd[W_DIM]

    group_dict = util_deconv_comm.calculate_group(
        shape_out_backprop,
        shape_res,
        shape_filters,
        groups,
        filters_dtype,
        filter_ori_format
    )

    util_deconv_comm.check_conv2dbp_input_params(
        shape_filters,
        shape_out_backprop,
        shape_res,
        strides,
        pads,
        dilations,
        filters_dtype,
        out_backprop_dtype,
        res_dtype,
        kernel_name=kernel_name,
        group_dict=group_dict
    )

    pads = util_deconv_comm.get_padlist(
        pads, shape_res, strides, shape_filters, dilations
    )

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    res = tbe.conv2d_backprop_input(filter, out_backprop, shape_filters, shape_res, para_dict=para_dict)

    return res


@para_check.check_input_type(
    (list, tuple),
    (list, tuple),
    (list, tuple),
    (list, tuple),
    (str, list, tuple),
    (list, tuple),
    str,
    str,
    str,
    str,
    (dict, para_check.NONE_TYPE)
)
def _conv2d_backprop_input_cce(  # pylint: disable=R0913,R0914
    shape_filter,
    shape_out_backprop,
    input_sizes,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    filter_dtype="float16",
    out_backprop_dtype="float16",
    res_dtype="float16",
    kernel_name="conv2d_backprop_input_cce",
    group_dict=None
):
    """
    Topi interface of conv2d backprop input

    Parameters:
    ----------
    shape_filter: The shape of filter.
        4-D with shape [batch, channels, height, weight].

    shape_out_backprop: The shape of gradients.
        4-D with shape [batch, channels, height, weight].

    input_sizes: The shape of feature map.
        4-D with shape [batch, channels, height, weight].

    strides: A tuple/list of ints. The stride of the sliding window.

    pads: "SAME"or"VALID" indicating the type of pads algorithm to use,
        or tuple/list.

    dilations: An optional tuple/list of ints. Default to (1, 1, 1, 1).

    filter_dtype: The dtype of filter data. Default to float16.

    out_backprop_dtype: The dtype of gradients data. Default to float16.

    res_dtype: The dtype of result(De/Dx) data. Default to float16.

    kernel_name: Cce kernel name. Default to "conv2d_backprop_input_cce".

    group_dict: The params of group_dict.

    Returns: None
    ----------
    """


    res = util_deconv_comm.check_conv2dbp_input_params(
        shape_filter,
        shape_out_backprop,
        input_sizes,
        strides,
        pads,
        dilations,
        filter_dtype,
        out_backprop_dtype,
        res_dtype,
        kernel_name=kernel_name,
        group_dict=group_dict
    )
    (
        shape_filter,
        shape_out_backprop,
        input_sizes,
        strides,
        pads,
        dilations,
        filter_dtype,
        out_backprop_dtype,
        res_dtype,
        kernel_name,
    ) = res

    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_channel, filter_h, filter_w = shape_filter

    _, dy_k0, _ = tbe_platform.CUBE_MKN[out_backprop_dtype]["mac"]
    _, w_k0, w_n0 = tbe_platform.CUBE_MKN[filter_dtype]["mac"]
    shape_dedy = (
        dedy_batch,
        util_deconv_comm.ceil(dedy_channel, dy_k0),
        dedy_h,
        dedy_w,
        dy_k0,
    )

    g_extend = group_dict.get(util_deconv_comm.GroupDictKeys.g_extend)
    dx_c1_extend = group_dict.get(util_deconv_comm.GroupDictKeys.dx_c1_extend)
    dy_c1_extend = group_dict.get(util_deconv_comm.GroupDictKeys.dy_c1_extend)
    groups = group_dict.get(util_deconv_comm.GroupDictKeys.groups)

    if filter_dtype == "int8" and out_backprop_dtype == "int8":
        shape_filter_frac = (
            g_extend * dy_c1_extend * filter_h * filter_w,
            dx_c1_extend,
            w_n0,
            w_k0,
        )
    else:
        shape_filter_frac = (
            # (GCi1HkWk, Co1, Co0, Ci0); filter_placehold is same to conv2d_forward's filter
            g_extend * dx_c1_extend * filter_h * filter_w,
            dy_c1_extend,
            w_n0,
            w_k0
        )
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)

    filter_frac = tvm.placeholder(shape_filter_frac, name="filter", dtype=filter_dtype)

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    dedx = tbe.conv2d_backprop_input(filters=filter_frac,
                                     out_backprop=dedy,
                                     filter_sizes=shape_filter,
                                     input_sizes=input_sizes,
                                     para_dict=para_dict)
    tensor_list = [filter_frac, dedy, dedx]

    with tvm.target.cce():
        sch = tbe.auto_schedule(dedx)

    config = {"name": kernel_name, "tensor_list": tensor_list}

    tbe.build(sch, config)
