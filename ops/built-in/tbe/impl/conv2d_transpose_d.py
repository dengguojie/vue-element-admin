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
conv2d_transpose_d
"""
from impl.util import util_deconv_comm
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000

# the shape dim limited
INOUT_HW_MIN = 1
INOUT_HW_MAX = 4096
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255
STRIDE_HW_MIN = 1
STRIDE_HW_MAX = 63
DILATION_HW_MIN = 1
DILATION_HW_MAX = 255
CONV1D_W_MAX = 2147483647

# 2 means L1 enable
L1FUSION_INPUT_CTR = 2

# position index
N_DIM = 0
H_DIM = 2
W_DIM = 3


def _check_attr_range(attr_name, attr_value, attr_min, attr_max):
    """
    check the value in the range: [attr_min, attr_max]
    """
    if attr_value < attr_min or attr_value > attr_max:
        args_dict = {
            "errCode": "E60011",
            "range": "[{},{}]".format(attr_min, attr_max),
            "attr_name": attr_name,
            "value": attr_value
        }
        raise RuntimeError(args_dict,
                           error_manager.get_error_message(args_dict))


def _check_param(  # pylint: disable=invalid-name,R0913,R0914,W0613
    x,
    filter,
    y,
    input_size,
    strides,
    pads,
    dilations,
    data_format,
    offset_x):
    """
    the params check of conv2d_transpose_d
    """
    ori_shape_x = x.get("ori_shape")
    ori_shape_filters = filter.get("ori_shape")
    ori_shape_res = y.get("ori_shape")
    x_dtype = x.get("dtype").lower()
    ori_format_x = x.get("ori_format")
    ori_format_filters = filter.get("ori_format")
    ori_format_res = y.get("ori_format")

    if list(input_size) != list(ori_shape_res):
        dict_args = {}
        dict_args["errCode"] = "E65007"
        dict_args["param1"] = "input_size"
        dict_args["param2"] = "ori_shape of y"
        dict_args["actual_value"] = "{}, {}".format(input_size, ori_shape_res)
        raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    # only quantify with offset
    if x_dtype != "int8" and offset_x != 0:
        reason = "when x_dtype is fp16, offset_x must be 0"
        args_dict = {"errCode": "E60108", "reason": reason}
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))

    if len(strides) == CONV_BACKPROP_SHAPE_DIM:
        h_index = data_format.find("H")
        w_index = data_format.find("W")
        strides = [strides[h_index], strides[w_index]]

    shape_filters = util_deconv_comm.get_filter_shape(ori_format_filters, ori_shape_filters)
    shape_x = util_deconv_comm.get_shape_out_backprop(ori_format_x, ori_shape_x)
    shape_res = util_deconv_comm.get_shape_res(ori_format_res, ori_shape_res)
    dilations = util_deconv_comm.get_shape_dilation(ori_format_x, dilations)

    if dilations[0] != 1 or dilations[1] != 1:
        args_dict = {
            "errCode": "E60023",
            "dilation_n": dilations[0],
            "dilation_c": dilations[1]
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))

    _check_attr_range("the h of filter", shape_filters[2], FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range("the w of filter", shape_filters[3], FILTER_HW_MIN, FILTER_HW_MAX)

    _check_attr_range("the h of stride", strides[0], STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range("the w of stride", strides[1], STRIDE_HW_MIN, STRIDE_HW_MAX)

    _check_attr_range("the h of dilations", dilations[2], DILATION_HW_MIN, DILATION_HW_MAX)
    _check_attr_range("the w of dilations", dilations[3], DILATION_HW_MIN, DILATION_HW_MAX)

    fmap_h, fmap_w = shape_res[2:]
    dedy_h, dedy_w = shape_x[2:]
    filter_h, filter_w = shape_filters[2:]
    filter_h_dilation = (filter_h - 1)*dilations[2] + 1
    filter_w_dilation = (filter_w - 1)*dilations[3] + 1
    pads = util_deconv_comm.get_padlist(pads, shape_res, strides, shape_filters, dilations)
    fmap_h_pad = fmap_h + pads[0] + pads[1]
    fmap_w_pad = fmap_w + pads[2] + pads[3]

    if ((fmap_h_pad - filter_h_dilation)//strides[0] + 1) != dedy_h:
        args_dict = {"errCode": "E60024"}
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
    if ((fmap_w_pad - filter_w_dilation)//strides[1] + 1) != dedy_w:
        args_dict = {"errCode": "E60025"}
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))

    inout_limit_min = INOUT_HW_MIN
    inout_limit_max = INOUT_HW_MAX
    if fmap_h_pad == 1 and filter_h_dilation == 1 and strides[0] == 1:
        inout_limit_max = CONV1D_W_MAX

    _check_attr_range("the h of fmap(output)", fmap_h, inout_limit_min, inout_limit_max)
    _check_attr_range("the w of fmap(output)", fmap_w, inout_limit_min, inout_limit_max)

    _check_attr_range("the h of dedy(input) after expands", dedy_h * strides[0], inout_limit_min, inout_limit_max)

    if filter_h == 1 and filter_w == 1:
        _check_attr_range("the w of dedy after expands",
                          dedy_w * strides[0] * strides[1], inout_limit_min, inout_limit_max)
    else:
        _check_attr_range("the w of dedy after expands",
                          dedy_w * strides[1], inout_limit_min, inout_limit_max)


def check_supported(x,
                    filter,
                    bias,
                    offset_w,
                    y,
                    input_size,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NHWC",
                    output_padding=(0, 0, 0, 0),
                    offset_x=0,
                    kernel_name="conv2d_transpose_d"):
    """
    the h and w of x or y must be in [1, 4096]
    the h and w of filter or dilations must be in [1, 255]
    the h and w of strides must be in [1, 63]
    the n and c of dilations must be 1
    the h and w must meet: 
       hi - (hk - 1)*dk + 1 + padh // strideh = ho
       wi - (wk - 1)*wk + 1 + padw // stridew = wo
    """
    try:
        _check_param(x, filter, y, input_size, strides, pads, dilations, data_format, offset_x)
        return True, ""
    except Exception as e:
        reason = e.args[1]
        return False, reason


def _cal_min_l1space(x,  # pylint: disable=invalid-name
                     weight, y, strides, dilations, pads):
    """
    cal the mini l1space using in lxfusion
    """
    def _cal_al1_size():
        w_value = shape_x[3] * strides[1]
        if shape_res[3] > c0_size:
            h_value_max = filter_h_dilation + 1
        elif c0_size % shape_res[3] == 0:
            h_value_max = filter_h_dilation + c0_size // shape_res[3] - 1
        else:
            h_value_max = filter_h_dilation + c0_size // shape_res[3] + 1
        a_l1_size = h_value_max * w_value * c0_size_k * \
                    util_deconv_comm.BIT_RATIO_DICT.get(filters_dtype)
        return a_l1_size

    shape_filters = util_deconv_comm.get_filter_shape(
        weight.get("ori_format"), weight.get("ori_shape")
    )
    shape_x = util_deconv_comm.get_shape_out_backprop(
        x.get("ori_format"), x.get("ori_shape"))
    shape_res = util_deconv_comm.get_shape_res(
        y.get("ori_format"), y.get("ori_shape"))
    filters_dtype = weight.get("dtype")

    c0_size = tbe_platform.C0_SIZE
    c0_size_k = tbe_platform.CUBE_MKN[filters_dtype]['mac'][1]
    filter_h_dilation = (shape_filters[2] - 1) * dilations[0] + 1

    bl1_size = shape_filters[2] * shape_filters[3] * c0_size * c0_size_k * \
               util_deconv_comm.BIT_RATIO_DICT.get(filters_dtype)
    al1_size = 0
    if (list(pads) != [0, 0, 0, 0] or list(shape_filters[2:]) != [1, 1]) \
            and list(strides) != [1, 1]:
        al1_size = _cal_al1_size()
    return al1_size + bl1_size


def get_op_support_info(x,  # pylint: disable=invalid-name,R0913,R0914,W0613
                        filter,
                        bias,
                        offset_w,
                        y,
                        input_size,
                        strides,
                        pads,
                        dilations=(1, 1, 1, 1),
                        groups=1,
                        data_format="NHWC",
                        offset_x=0,
                        kernel_name="conv2d_transpose_d"):
    """
    get the conv2d_transpose_d split
    """
    format_x = x.get("format")
    dtype_x = x.get("dtype")
    h_pos = data_format.find("H")
    w_pos = data_format.find("W")

    shape_filters = util_deconv_comm.get_filter_shape(filter.get("ori_format"), filter.get("ori_shape"))
    head_overlap_h = -1 if (shape_filters[2] == 1 and strides[h_pos] == 1) else 0
    tail_overlap_h = head_overlap_h
    head_overlap_w = -1 if (shape_filters[3] == 1 and strides[w_pos] == 1) else 0
    tail_overlap_w = head_overlap_w

    # input/output Serial， axis Serial, (headoverlap, tailoverlap, 0 means with overlap, -1 means without it)
    if format_x == "NC1HWC0":
        axis_split_matrix = [
            # cut N
            [util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
             util_select_op_base.SplitOutput([0, [0]])],
            # cut H
            [util_select_op_base.SplitInput([0, [2], [head_overlap_h], [tail_overlap_h]]),
             util_select_op_base.SplitOutput([0, [2]])],
            # cut W
            [util_select_op_base.SplitInput([0, [3], [head_overlap_w], [tail_overlap_w]]),
             util_select_op_base.SplitOutput([0, [3]])],
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

    min_l1space = _cal_min_l1space(x, filter, y, strides, dilations, pads)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT,
    para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.KERNEL_NAME,
)
def conv2d_transpose_d(  # pylint: disable=R0913,R0914,W0613,W0622,C0103
    x,
    filter,
    bias,
    offset_w,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    output_padding=(0, 0, 0, 0),
    offset_x=0,
    kernel_name="conv2d_transpose_d",
):
    """
    algorithm: conv2d_transpose_d

    Parameters
    ----------
    x: dict with keys(shape and dtype)
        The shape of gradients.

    filter: dict with keys(shape and dtype)
        input filter tensor.

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
        Input offset_w tensor.

    y: dict with keys(shape and dtype)
       conv2d_transpose_d output tensor, dtype must be assigned.

    input_size: tuple/list of 4 integers
        The shape of feature map. 4-D with shape [batch, height, width, channels]
        or [batch, channels, height, filter].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_transpose_d. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_transpose_d. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    output_padding: tuple/list of 4 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 1.

    kernel_name: str
        kernel name. Default to "conv2d_transpose_d".

    Returns
    -------
    None
    """

    ori_shape_x = x.get("ori_shape")
    ori_shape_filters = filter.get("ori_shape")
    ori_shape_res = y.get("ori_shape")

    x_dtype = x.get("dtype")
    filters_dtype = filter.get("dtype")
    res_dtype = y.get("dtype")

    ori_format_x = x.get("ori_format")
    ori_format_filters = filter.get("ori_format")
    ori_format_res = y.get("ori_format")

    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(
        ori_shape_filters,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )
    para_check.check_shape_rule(
        ori_shape_x,
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

    shape_x_5hd = x.get("shape")
    shape_x = util_deconv_comm.get_shape_out_backprop(ori_format_x, ori_shape_x)
    shape_x = list(shape_x)
    shape_x[N_DIM] = shape_x_5hd[N_DIM]
    shape_x[H_DIM] = shape_x_5hd[H_DIM]
    shape_x[W_DIM] = shape_x_5hd[W_DIM]

    shape_res = util_deconv_comm.get_shape_res(ori_format_res, ori_shape_res)

    dilations = util_deconv_comm.get_shape_dilation(ori_format_x, dilations)

    bias_flag = bias is not None

    _conv2d_transpose_cce(
        shape_filters,
        shape_x,
        shape_res,
        strides,
        pads,
        dilations=dilations,
        groups=groups,
        filter_dtype=filters_dtype,
        x_dtype=x_dtype,
        res_dtype=res_dtype,
        bias=bias_flag,
        offset_x=offset_x,
        kernel_name=kernel_name,
        ori_format_filters=ori_format_filters,
    )


@tbe_platform.fusion_manager.register("conv2d_transpose_d")
def conv2d_transpose_d_compute(  # pylint: disable=R0913,R0914,W0613,C0103,W0622
    x,
    filter,
    bias,
    offset_w,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    output_padding=(0, 0, 0, 0),
    offset_x=0,
    kernel_name="conv2d_transpose_d",
):
    """
    used for fusion
    Parameters
    ----------
    x: dict with keys(shape and dtype)
        The shape of gradients.

    filter: dict with keys(shape and dtype)
        input filter tensor

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
        Input offset_w tensor.

    y: dict with keys(shape and dtype)
       conv2d_transpose_d output tensor, dtype must be assigned.

    input_size: tuple/list of 4 integers
        The shape of feature map. 4-D with shape [batch, height, filter, channels]
        or [batch, channels, height, filter].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_transpose_d. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_transpose_d. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Defaults to "NHWC".

    output_padding: tuple/list of 4 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 1.

    kernel_name: str
        kernel name. Default to "conv2d_transpose_d".

    Returns
    -------
    None
    """
    ori_shape_filter = [i.value for i in filter.op.attrs["ori_shape"]]
    ori_shape_x = [i.value for i in x.op.attrs["ori_shape"]]
    shape_x_5hd = [i.value for i in x.shape]
    ori_shape_res = [i for i in y["ori_shape"]]

    filter_dtype = filter.dtype
    x_dtype = x.dtype
    res_dtype = y["dtype"]

    ori_format_filter = filter.op.attrs["ori_format"]
    ori_format_x = x.op.attrs["ori_format"]
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

    if filter_dtype == "int8":
        ori_shape_filter = util_deconv_comm.exchange_filter_nc_axis(
            ori_format_filter, ori_shape_filter, groups
        )

    shape_filter = util_deconv_comm.get_filter_shape(
        ori_format_filter, ori_shape_filter
    )
    shape_x = util_deconv_comm.get_shape_out_backprop(ori_format_x, ori_shape_x)
    shape_x = list(shape_x)
    shape_x[N_DIM] = shape_x_5hd[N_DIM]
    shape_x[H_DIM] = shape_x_5hd[H_DIM]
    shape_x[W_DIM] = shape_x_5hd[W_DIM]
    shape_res = util_deconv_comm.get_shape_res(ori_format_res, ori_shape_res)
    dilations = util_deconv_comm.get_shape_dilation(ori_format_x, dilations)

    group_dict = util_deconv_comm.calculate_group(
        shape_x,
        shape_res,
        shape_filter,
        groups,
        filter_dtype,
        ori_format_filter
    )
    util_deconv_comm.check_conv2dbp_input_params(
        shape_filter,
        shape_x,
        shape_res,
        strides,
        pads,
        dilations,
        filter_dtype,
        x_dtype,
        res_dtype,
        kernel_name,
        group_dict=group_dict
    )

    pads = util_deconv_comm.get_padlist(
        pads, shape_res, strides, shape_filter, dilations
    )

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "tensor_bias": bias,
        "offset_x": offset_x,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "is_fusion_flag": True
    }

    res = tbe.conv2d_backprop_input(filter, x, shape_filter, shape_res, para_dict=para_dict)

    return res


@para_check.check_input_type(
    (list, tuple),
    (list, tuple),
    (list, tuple),
    (list, tuple),
    (str, list, tuple),
    (list, tuple),
    int,
    str,
    str,
    str,
    bool,
    int,
    str,
    str
)
def _conv2d_transpose_cce(
    shape_filter,  # pylint: disable=R0913, R0914
    shape_x,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    filter_dtype="float16",
    x_dtype="float16",
    res_dtype="float16",
    bias=False,
    offset_x=0,
    kernel_name="conv2d_transpose_cce",
    ori_format_filters="NCHW"
):
    """
    Topi interface of conv2d_transpose_d

    Parameters:
    ----------
    shape_filter: The shape of filter.
        4-D with shape [batch, channels, height, width].

    shape_x: The shape of gradients.
        4-D with shape [batch, channels, height, width].

    input_size: The shape of feature map.
        4-D with shape [batch, channels, height, width].

    strides: A tuple/list of ints.
        The stride of the sliding window.

    pads: "SAME"or"VALID", indicating the type of pads algorithm to use,
        or tuple/list .

    dilations: An optional tuple/list of ints. Default to (1, 1, 1, 1).

    groups : The params of group_dict. Default to 1.

    filter_dtype: The dtype of filter data. Default to float16.

    x_dtype: The dtype of gradients data. Default to float16.

    res_dtype: The dtype of result(De/Dx) data. Default to float16.

    bias: False: no bias, True: have bias. Default to 0.

    kernel_name: Cce kernel name. Default to "conv2d_transpose_cce".

    ori_format_weight : The original format of filter. Default to NCHW.

    Returns: None
    ----------
    """

    def _ceil(x_1, x_2):
        if x_2 == 0:
            dict_args = {}
            dict_args['errCode'] = "E60108"
            dict_args['reason'] = "Division by zero"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        return (x_1 + x_2 - 1) // x_2

    if filter_dtype == "int8" and x_dtype == "int8":
        if shape_filter[0] % groups != 0:
            args_dict = {
                "errCode": "E60108",
                "reason": "batch of weight % groups must be 0",
            }
            raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
        shape_filter = [
            shape_filter[1] * groups,
            shape_filter[0] // groups,
            shape_filter[2],
            shape_filter[3],
        ]

    group_dict = util_deconv_comm.calculate_group(
        shape_x,
        input_size,
        shape_filter,
        groups,
        filter_dtype,
        ori_format_filters
    )
    res = util_deconv_comm.check_conv2dbp_input_params(
        shape_filter,
        shape_x,
        input_size,
        strides,
        pads,
        dilations,
        filter_dtype,
        x_dtype,
        res_dtype,
        kernel_name,
        group_dict=group_dict
    )

    (
        shape_filter,
        shape_x,
        input_size,
        strides,
        pads,
        dilations,
        filter_dtype,
        x_dtype,
        res_dtype,
        kernel_name,
    ) = res

    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_x
    filter_batch, filter_channel, filter_h, filter_w = shape_filter

    _, dy_k0, _ = tbe_platform.CUBE_MKN[x_dtype]["mac"]
    _, w_k0, w_n0 = tbe_platform.CUBE_MKN[filter_dtype]["mac"]
    shape_dedy = (dedy_batch, _ceil(dedy_channel, dy_k0), dedy_h, dedy_w, dy_k0)

    g_extend = group_dict.get(util_deconv_comm.GroupDictKeys.g_extend)
    dx_c1_extend = group_dict.get(util_deconv_comm.GroupDictKeys.dx_c1_extend)
    dy_c1_extend = group_dict.get(util_deconv_comm.GroupDictKeys.dy_c1_extend)
    groups = group_dict.get(util_deconv_comm.GroupDictKeys.groups)

    if filter_dtype == "int8" and x_dtype == "int8":
        shape_filter_frac = (
            g_extend * dy_c1_extend * filter_h * filter_w,
            dx_c1_extend,
            w_n0,
            w_k0,
        )
    else:
        shape_filter_frac = (
            g_extend * dx_c1_extend * filter_h * filter_w,
            dy_c1_extend,
            w_k0,
            w_n0,
        )
    tensor_dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=x_dtype)

    tensor_filter_frac = tvm.placeholder(
        shape_filter_frac, name="filter", dtype=filter_dtype
    )

    if bias:
        input_channel = util_deconv_comm.align(input_size[1], w_n0)
        tensor_bias = tvm.placeholder(
            (input_channel,), name="tensor_bias", dtype=res_dtype
        )
    else:
        tensor_bias = None

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "tensor_bias": tensor_bias,
        "offset_x": offset_x,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    dedx = tbe.conv2d_backprop_input(filters=tensor_filter_frac,
                                     out_backprop=tensor_dedy,
                                     filter_sizes=shape_filter,
                                     input_sizes=input_size,
                                     para_dict=para_dict)

    if bias:
        tensor_list = [tensor_dedy, tensor_filter_frac, tensor_bias, dedx]
    else:
        tensor_list = [tensor_dedy, tensor_filter_frac, dedx]

    with tvm.target.cce():
        sch = tbe.auto_schedule(dedx)

    config = {"name": kernel_name, "tensor_list": tensor_list}

    tbe.build(sch, config)
