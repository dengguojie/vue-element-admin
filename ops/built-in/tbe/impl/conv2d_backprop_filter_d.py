# Copyright 2019 Huawei Technologies Co., Ltd
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
conv2d_backprop_filter_d
"""
from impl.util import util_select_op_base
from impl.util import util_deconv_comm
from impl.util.platform_adapter import error_manager
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_cube

# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4

# the dim of strides in conv_backprop must be 2
STRIDES_SHAPE_DIM = 2

# the min x or y dim for cube mul
C0 = 16
# fmapH, must be in [1,200000]
FMAP_H_MAX = 200000
# fmapW, must be in [1,4096]
FMAP_W_MAX = 4096
FMAP_HW_MIN = 1

# DeDyH must be in [1,200000]
DEDY_H_MAX = 200000
# DeDyW must be in [1,4096]
DEDY_W_MAX = 4096
DEDY_HW_MIN = 1

# filterH, filterW must be in [1,255]
FILTER_HW_MAX = 255
FILTER_HW_MIN = 1

# stride must be in [1,63]
STRIDE_HW_MAX = 63
STRIDE_HW_MIN = 1

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000

# the max size is 2**63-1
DATA_SIZE_MAX = 9223372036854775807

L1FUSION_INPUT_CTR = 2

# the bytes length of several dtype
BIT_RATIO_DICT = {
    "int32": 4,
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "int8": 1,
    "uint4": 0.5,
    "int4": 0.5,
    "bfloat16": 2,
}

# pads valid mode to be [0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0]
# If pads is string , only support "SAME" or "VALID"
PADDING_SUPPORT = ("SAME", "VALID")
# conv1d situation support w not larger than 2^31-1
CONV1D_MAX_W = 2147483647


def _align(input_x, input_y):
    if input_y == 0:
        dict_args = {
            "errCode": "E60108",
            "reason": "Division by zero"
        }
        raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
    return (input_x + input_y - 1) // input_y * input_y


def _cal_min_l1space(
    shape_x,
    shape_out_backprop,
    filter_sizes,
    strides,
    dilations,
    pads,
    dtype="float16"
):
    """
    cal the mini l1space using in lxfusion
    Parameters
    ----------
    shape_x: tuple/list of 4 integers
    shape_out_backprop: tuple/list of 4 integers
    filter_sizes: filter_sizes
    strides: tuple/list of 2 integers
    dilations: tuple/list of 4 integers
    pads: tuple/list of 4 integers or string

    Returns
    -------
    bl1_min_byte: int
    """
    # Waiting for FE support fp32, need to be deleted later
    dtype = "float16"
    filter_sizes = list(filter_sizes)
    strides = list(strides)
    dilations = list(dilations)
    _, _, fmap_h, fmap_w = shape_x
    dedy_w = shape_out_backprop[3]
    _, _, filter_h, filter_w = filter_sizes
    stride_h, stride_w = strides
    _, _, dilation_h, dilation_w = dilations
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    if pads == "VALID":
        pad_up = 0
        pad_down = 0
    elif pads == "SAME":
        pad_h = max(
            _align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h,
            0
        )
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
    else:
        pad_up, pad_down = pads[0:2]
    fmap_h_padding = fmap_h + pad_up + pad_down

    if fmap_h_padding == 1 and filter_h_dilation == 1 and stride_h == 1:
        kl1_min = (C0 - 1) * stride_w + filter_w_dilation
    else:
        kl1_min = fmap_w

    if dedy_w % C0 == 0:
        bl1_min_byte = filter_h_dilation * kl1_min * C0 * BIT_RATIO_DICT.get(dtype)
    else:
        bl1_min_byte = (filter_h_dilation + stride_h) * kl1_min * C0 * BIT_RATIO_DICT.get(dtype)

    return bl1_min_byte


def _get_shape_by_format(ori_format, shape, param_name, support_hwcn=False):
    """
    format shape to NCHW
    Parameters
    ----------
    ori_format: string
        origin format
    shape: list or tuple of 4 integers
    param_name: string
    support_hwcn: bool
    Returns
    -------
    res: list of 4 integers
        formatted shape of NCHW
    """

    if ori_format == "NHWC":
        res = [shape[0], shape[3], shape[1], shape[2]]
    elif ori_format == "NCHW":
        res = shape[:]
    else:
        if support_hwcn and ori_format == "HWCN":
            res = [shape[3], shape[2], shape[0], shape[1]]
        else:
            format_list = "[NCHW, NHWC, HWCN]" if support_hwcn else "[NCHW, NHWC]"
            dict_args = {
                "errCode": "E60008",
                "param_name": param_name,
                "expected_format_list": format_list,
                "format": ori_format
            }
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
    return res


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
def get_op_support_info(
    x,  # pylint: disable=invalid-name
    out_backprop,
    y,
    filter_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    kernel_name="conv2d_backprop_filter",
):
    """
    get the conv2d_backprop_filter split info

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input feature map tensor.

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input weight tensor.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype)
        output tensor, dtype must be assigned.

    filter_size: tuple/list of 4 integers

    strides: tuple/list of 2 integers

    pads: tuple/list of 4 integers or string

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_filter. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_filter".

    Returns
    -------
    split info, split axis and min l1 space
    """

    format_x = x.get("format")
    dtype_x = x.get("dtype")
    axis_split_matrix = None
    axis_reduce_list = None
    if format_x == "NC1HWC0":
        # only Cout1 can be cut without overlap
        axis_split_matrix = [
            [util_select_op_base.SplitInput([1, [1], [-1], [-1]]),
             util_select_op_base.SplitOutput([0, [1]])]
        ]
        axis_reduce_list = [
            [util_select_op_base.ReduceInput([0, [0]], [1, [0]]),
             util_select_op_base.ReduceOutput([0, 1, False])]
        ]

    ori_format_x = x.get("ori_format")
    ori_shape_x = x.get("ori_shape")
    ori_format_out_backprop = out_backprop.get("ori_format")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_format_y = y.get("ori_format")
    ori_shape_y = y.get("ori_shape")
    x_shape = _get_shape_by_format(ori_format_x, ori_shape_x, "x")
    shape_out = _get_shape_by_format(ori_format_out_backprop, ori_shape_out_backprop, "out_backprop")
    filter_shape = _get_shape_by_format(ori_format_y, ori_shape_y, "y", True)
    if len(strides) == 4:
        h_index = data_format.find("H")
        w_index = data_format.find("W")
        strides = [strides[h_index], strides[w_index]]
    min_l1space = _cal_min_l1space(x_shape, shape_out, filter_shape, strides, dilations, pads, dtype_x)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)
    return op_cal_info_in_json


def check_supported(x,
                    out_backprop,
                    y,
                    filter_size,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NHWC",
                    kernel_name="conv2d_backprop_filter"):
    """
    check the op support situation:

    | Name             | Field    | Scope
    -------------------|----------|--------------
    | x                | H or W   | [1, 4096]
    -------------------|----------|--------------
    | out_backprop     | H or W   | [1, 4096]
    -------------------|----------|--------------
    | filter_size      | H or W   | [1, 255]
    -------------------|----------|--------------
    | y(filter)        | H or W   | [1, 255]
    -------------------|----------|--------------
    | Stride           | H or W   | [1, 63]
    -------------------|----------|--------------
    | Dilation         | H or W   | [1, 255]

    In Ascend910, out_backprop's H and W not support 1
    when fmap_h + pad_top + pad_bottom != (filter_height - 1) * dilation_h + 1

    batch_x == batch_out_backprop

    batch_filter == channel_out_backprop

    channel_filter == channel_x * groups

    out_backprop_height == (fmap_height + pad_top + pad_bottom -
                          (dilation_h * (filter_height - 1) + 1))
                           / stride_h + 1

    out_backprop_width == (fmap_width + pad_left + pad_right -
                         (dilation_w * (filter_width - 1) + 1))
                          / stride_w + 1
    """
    shape_x = x.get("ori_shape")
    dynamic_flag = any([i < 0 for i in shape_x])
    if dynamic_flag:
        return True, ""
    try:
        res = _check_shape_and_format(x, out_backprop, y, filter_size, strides,
                                      pads, dilations, groups, data_format)
        [
            shape_x, shape_out_backprop, shape_res, strides, dilations,
            x_dtype, out_backprop_dtype, res_dtype
        ] = res

        check_conv2dbp_filter_params(shape_x, shape_out_backprop, shape_res,
                                     strides, pads, dilations, groups, x_dtype,
                                     out_backprop_dtype, res_dtype,
                                     kernel_name)
        return True, ""
    except RuntimeError as e:
        reason = e.args[1]
        return False, reason


def _check_shape_and_format(  # pylint: disable=W0622,C0103,R0913,R0914
    x,
    out_backprop,
    y,
    filter_size,
    strides,
    pads,
    dilations,
    groups,
    data_format
    ):

    """
    check the shape dims, format and get NCHW format shape
    """

    def _check_inputs_rules():
        if (not isinstance(ori_shape_out_backprop, (tuple, list))) or len(
            ori_shape_out_backprop
        ) != 4:
            dict_args = dict()
            dict_args["errCode"] = "E60107"
            dict_args["param_name"] = "out_backprop"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

        if (not isinstance(ori_shape_x, (tuple, list))) or len(ori_shape_x) != 4:
            dict_args = dict()
            dict_args["errCode"] = "E60107"
            dict_args["param_name"] = "x"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

        if (not isinstance(ori_shape_res, (tuple, list))) or len(ori_shape_res) != 4:
            dict_args = dict()
            dict_args["errCode"] = "E60107"
            dict_args["param_name"] = "y"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

        if len(strides) != 2:
            dict_args = dict()
            dict_args["errCode"] = "E60107"
            dict_args["param_name"] = "strides"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

        if len(filter_size) != 4:
            dict_args = dict()
            dict_args["errCode"] = "E60107"
            dict_args["param_name"] = "filter_size"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

        if len(dilations) != 4:
            dict_args = dict()
            dict_args["errCode"] = "E60107"
            dict_args["param_name"] = "dilations"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        if list(filter_size) != list(ori_shape_res):
            dict_args = {}
            dict_args["errCode"] = "E64002"
            dict_args["param1"] = "filter_size"
            dict_args["param2"] = "ori_shape of y"
            dict_args["actual_value"] = "{}, {}".format(filter_size, ori_shape_res)
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    def _calcute_input_shape():
        if ori_format_x == "NHWC":
            x_shape = (ori_shape_x[0], ori_shape_x[3], ori_shape_x[1], ori_shape_x[2])
        elif ori_format_x == "NCHW":
            x_shape = ori_shape_x
        else:
            dict_args = {}
            dict_args["errCode"] = "E60008"
            dict_args["param_name"] = "x"
            dict_args["expected_format_list"] = "[{}, {}]".format("NHWC", "NCHW")
            dict_args["format"] = ori_format_x
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

        if ori_format_out_backprop == "NCHW":
            shape_out = ori_shape_out_backprop
        elif ori_format_out_backprop == "NHWC":
            shape_out = (
                ori_shape_out_backprop[0],
                ori_shape_out_backprop[3],
                ori_shape_out_backprop[1],
                ori_shape_out_backprop[2],
            )
        else:
            dict_args = {}
            dict_args["errCode"] = "E60008"
            dict_args["param_name"] = "out_backprop"
            dict_args["expected_format_list"] = "[{}, {}]".format("NHWC", "NCHW")
            dict_args["format"] = ori_format_out_backprop
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        return x_shape, shape_out

    ori_shape_x = x.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = y.get("ori_shape")

    x_dtype = x.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y.get("dtype")

    ori_format_x = x.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")
    ori_format_res = y.get("ori_format")

    if len(strides) == 4:
        h_index = data_format.find("H")
        w_index = data_format.find("W")
        strides = [strides[h_index], strides[w_index]]

    _check_inputs_rules()
    dilations = _get_shape_dilation(data_format, dilations)
    shape_x, shape_out_backprop = _calcute_input_shape()

    if ori_format_res == "NCHW":
        shape_res = ori_shape_res
    elif ori_format_res == "NHWC":
        shape_res = (
            ori_shape_res[0],
            ori_shape_res[3],
            ori_shape_res[1],
            ori_shape_res[2],
        )
    elif ori_format_res == "HWCN":
        shape_res = (
            ori_shape_res[3],
            ori_shape_res[2],
            ori_shape_res[0],
            ori_shape_res[1],
        )
    else:
        dict_args = {}
        dict_args["errCode"] = "E60008"
        dict_args["param_name"] = "y"
        dict_args["expected_format_list"] = "[{}, {}, {}]".format(
            "NHWC", "NCHW", "HWCN"
        )
        dict_args["format"] = ori_format_res
        raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
    return [shape_x,
            shape_out_backprop,
            shape_res,
            strides,
            dilations,
            x_dtype,
            out_backprop_dtype,
            res_dtype]


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
def conv2d_backprop_filter_d(
    x,
    out_backprop,
    y,
    filter_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    kernel_name="conv2d_backprop_filter",
    ):
    """
    algorithm: conv2d_backprop_filter

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input feature map tensor.

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype)
        input weight tensor.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype)
        output tensor, dtype must be assigned.

    filter_size: tuple/list of 4 integers
        The shape of filter. 4-D with shape [filter_height, filter_width, in_channels,
        out_channels] or [out_channels, filter_height, filter_width, in_channels] or
        [out_channels, in_channel, filter_height, filter_width].

    strides: tuple/list of 2 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_backprop_filter. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    kernel_name: str
        kernel name. Default to "conv2d_backprop_filter".

    Returns
    -------
    None
    """

    res = _check_shape_and_format(x,
                                  out_backprop,
                                  y,
                                  filter_size,
                                  strides,
                                  pads,
                                  dilations,
                                  groups,
                                  data_format)
    [shape_x,
     shape_out_backprop,
     shape_res,
     strides,
     dilations,
     x_dtype,
     out_backprop_dtype,
     res_dtype] = res

    _conv2d_backprop_filter_cce(
        shape_x,
        shape_out_backprop,
        shape_res,
        strides,
        pads,
        dilations,
        groups,
        x_dtype,
        out_backprop_dtype,
        res_dtype,
        kernel_name,
    )


def _get_shape_dilation(data_format, dilations):
    """
    Get result shape of NCHW from original shape
    :param ori_format_res:
    :param ori_shape_res:
    :return: result shape of NCHW
    """
    if data_format == "NCHW":
        shape_dilations = dilations
    elif data_format == "NHWC":
        shape_dilations = (dilations[0], dilations[3], dilations[1], dilations[2])
    else:
        dict_args = {}
        dict_args["errCode"] = "E60004"
        dict_args["param_name"] = "data_format"
        dict_args["expected_format_list"] = "[{}, {}]".format("NHWC", "NCHW")
        dict_args["format"] = data_format
        raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
    return shape_dilations


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
    str,
)
def check_conv2dbp_filter_params(
    shape_x,
    shape_out_backprop,
    filter_sizes,
    strides,
    pads,
    dilations,
    groups,
    x_dtype,
    out_backprop_dtype,
    res_dtype,
    kernel_name,
    ):
    """
    The params check function of conv2d_backprop_filter

    Parameters:
    ----------
    shape_x: The shape of feature map, which is 4-D [batch, channels, height, weight].

    shape_out_backprop: The shape of gradients, which is 4-D [batch, channels, height,
        weight].

    filter_sizes: The shape of filter, which is 4-D [batch, channels, height, weight].

    strides: The stride of the sliding window. A tuple/list of ints.

    pads: "SAME"or"VALID", indicating the type of pads algorithm to use, or tuple/list.

    dilations: An optional tuple/list of ints.

    groups : The number of filter's group. Default value is 1.

    x_dtype: Fmeature map data dtype.

    out_backprop_dtype: Gradients data dtype.

    res_dtype: Result(De/Dw) data dtype.

    kernel_name: Kernel name of cce.

    Returns : All transformed params.
    ----------
    """

    def _check_attr_range_dw(name, value, attr_min=None, attr_max=None):
        if not attr_min and not attr_max:
            return
        if not attr_min:
            if (not isinstance(value, int)) or value > attr_max:
                dict_args = {}
                dict_args["errCode"] = "E64001"
                dict_args["range"] = "(, {}]".format(attr_max)
                dict_args["attr_name"] = name
                dict_args["value"] = str(value)
                raise RuntimeError(
                    dict_args, error_manager.get_error_message(dict_args)
                )
        elif not attr_max:
            if (not isinstance(value, int)) or value < attr_min:
                dict_args = {}
                dict_args["errCode"] = "E64001"
                dict_args["range"] = "[{}, )".format(attr_min)
                dict_args["attr_name"] = name
                dict_args["value"] = str(value)
                raise RuntimeError(
                    dict_args, error_manager.get_error_message(dict_args)
                )
        elif (not isinstance(value, int)) or value > attr_max or value < attr_min:
            dict_args = {}
            dict_args["errCode"] = "E64001"
            dict_args["range"] = "[{},{}]".format(attr_min, attr_max)
            dict_args["attr_name"] = name
            dict_args["value"] = str(value)
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype:
            bit_ratio = BIT_RATIO_DICT.get(dtype)
        else:
            bit_ratio = BIT_RATIO_DICT.get("float16")
        if attr_value * bit_ratio > DATA_SIZE_MAX:
            dict_args = {}
            dict_args["errCode"] = "E60020"
            dict_args["attr_name"] = attr_name
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    def _need_change_hw():
        return fmap_w == 1 and filter_w == 1 and dedy_w == 1 and pad_left == 0 and pad_right == 0

    # First : Base check, Mainly required by interface appearance
    # ===========================================================
    # util check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(
        shape_x, CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM
    )
    para_check.check_shape_rule(
        shape_out_backprop,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )
    para_check.check_shape_rule(
        filter_sizes,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )
    para_check.check_shape_rule(
        strides, STRIDES_SHAPE_DIM, STRIDES_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM
    )

    def _check_attr_pads():
        # pads check
        if isinstance(pads, (tuple, list)) and len(pads) != CONV_BACKPROP_SHAPE_DIM:
            dict_args = dict()
            dict_args["errCode"] = "E60107"
            dict_args["param_name"] = "pads"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

        if isinstance(pads, str) and pads not in PADDING_SUPPORT:
            dict_args = {}
            dict_args["errCode"] = "E60021"
            dict_args["expected_pad_mode"] = str(PADDING_SUPPORT)
            dict_args["actual_pad_mode"] = str(pads)

            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    _check_attr_pads()

    # dilations check
    para_check.check_shape_rule(
        dilations,
        CONV_BACKPROP_SHAPE_DIM,
        CONV_BACKPROP_SHAPE_DIM,
        DEFAULT_MAX_SHAPE_NUM,
    )
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    _check_attr_range_dw("dilations's H", dilation_h, DILATION_MIN, DILATION_MAX)
    _check_attr_range_dw("dilations's W", dilation_w, DILATION_MIN, DILATION_MAX)
    if dilation_n != 1 or dilation_c != 1:
        dict_args = {}
        dict_args["errCode"] = "E60023"
        dict_args["dilation_n"] = str(dilation_n)
        dict_args["dilation_c"] = str(dilation_c)
        raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    # Second : Furture Check, Mainly required by SRS
    # ===========================================================
    # the relation limits between shape
    shape_x = list(shape_x)
    shape_out_backprop = list(shape_out_backprop)
    filter_sizes = list(filter_sizes)
    strides = list(strides)
    fmap_batch, fmap_channel, fmap_h, fmap_w = shape_x
    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_channel, filter_h, filter_w = filter_sizes
    stride_h, stride_w = strides

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    # groups check
    if fmap_channel % groups != 0:
        dict_args = {
            'errCode': "E60108",
            'reason': "fmap's channel must be a multiple of groups"
        }
        raise RuntimeError(dict_args,
                           error_manager.get_error_message(dict_args))

    if dedy_channel % groups != 0:
        dict_args = {
            'errCode': "E60108",
            'reason': "outbackprop's channel must be a multiple of groups"
        }
        raise RuntimeError(dict_args,
                           error_manager.get_error_message(dict_args))

    # pads compute
    if pads == "SAME":
        pad_w = _align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = _align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pads = [pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = PADDING_VAILD
    pads = list(pads)
    pad_up, pad_down, pad_left, pad_right = pads

    fmap_w_padding = fmap_w + pad_left + pad_right
    fmap_h_padding = fmap_h + pad_up + pad_down

    # exchange h and w will not change date in memory
    if _need_change_hw():
        shape_x = (fmap_batch, fmap_channel, fmap_w, fmap_h)
        shape_out_backprop = (dedy_batch, dedy_channel, dedy_w, dedy_h)
        filter_sizes = (filter_batch, filter_channel, filter_w, filter_h)
        strides = stride_w, stride_h
        stride_h, stride_w = stride_w, stride_h
        dilations = dilation_n, dilation_c, dilation_w, dilation_h
        fmap_h_padding, fmap_w_padding = fmap_w_padding, fmap_h_padding
        dedy_h, dedy_w = dedy_w, dedy_h
        fmap_h, fmap_w = fmap_w, fmap_h
        filter_h, filter_w = filter_w, filter_h
        filter_h_dilation, filter_w_dilation = filter_w_dilation, filter_h_dilation
        pad_left, pad_right, pad_up, pad_down = pads
        pads = pad_up, pad_down, pad_left, pad_right

    def _check_axis_hw():
        if fmap_batch != dedy_batch:
            dict_args = {}
            dict_args["errCode"] = "E64002"
            dict_args["param1"] = "x's N"
            dict_args["param2"] = "out_backprop's N"
            dict_args["actual_value"] = "{}, {}".format(fmap_batch, dedy_batch)
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        if dedy_channel != filter_batch:
            dict_args = {}
            dict_args["errCode"] = "E64002"
            dict_args["param1"] = "out_backprop's C"
            dict_args["param2"] = "Filter's N"
            dict_args["actual_value"] = "{}, {}".format(dedy_channel, filter_batch)
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        if fmap_channel != filter_channel * groups:
            dict_args = {}
            dict_args["errCode"] = "E64002"
            dict_args["param1"] = "x's C"
            dict_args["param2"] = "y's C"
            dict_args["actual_value"] = "{}, {}".format(fmap_channel, filter_channel)
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        if filter_w_dilation > fmap_w_padding:
            dict_args = dict()
            dict_args["errCode"] = "E60015"
            dict_args["w_of_x"] = str(fmap_w_padding)
            dict_args["w_of_filter"] = str(filter_w_dilation)
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        if filter_h_dilation > fmap_h_padding:
            dict_args = dict()
            dict_args["errCode"] = "E60014"
            dict_args["h_of_x"] = str(fmap_h_padding)
            dict_args["h_of_filter"] = str(filter_h_dilation)
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

        # Third : value check, Mainly required by the convolution rule
        if (
            (fmap_w - filter_w_dilation + pad_left + pad_right) // stride_w + 1
        ) != dedy_w:
            dict_args = {}
            dict_args["errCode"] = "E60025"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        if ((fmap_h - filter_h_dilation + pad_up + pad_down) // stride_h + 1) != dedy_h:
            dict_args = {}
            dict_args["errCode"] = "E60024"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    _check_axis_hw()

    # Fifth : check shape size, 64 bits limitation
    c0_size = tbe_platform.C0_SIZE
    fmap_size = fmap_batch * _align(fmap_channel, c0_size) * fmap_h * fmap_w
    dedy_size = dedy_batch * _align(dedy_channel, c0_size) * dedy_h * dedy_w
    filter_size = (
        _align(filter_batch, c0_size)
        * _align(filter_channel, c0_size)
        * filter_h
        * filter_w
    )
    _check_64bits_limitation("fmap_size", fmap_size, dtype=x_dtype)
    _check_64bits_limitation("dedy_size", dedy_size, dtype=out_backprop_dtype)
    _check_64bits_limitation("filter_size", filter_size, dtype=res_dtype)

    result = (
        shape_x,
        shape_out_backprop,
        filter_sizes,
        strides,
        pads,
        dilations,
        groups,
        x_dtype,
        out_backprop_dtype,
        res_dtype,
        kernel_name,
    )
    return result


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
    str,
)
def _conv2d_backprop_filter_cce(
    shape_x,
    shape_out_backprop,
    filter_sizes,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    x_dtype="float16",
    out_backprop_dtype="float16",
    res_dtype="float32",
    kernel_name="conv2d_backprop_filter_cce",
):
    """
    Topi interface of conv2d backprop filter

    Parameters:
    ----------
    shape_x: The shape of feature map. 4-D with shape [batch, channels, height, weight].

    shape_out_backprop: The shape of gradients. 4-D with shape [batch, channels, height,
        weight].

    filter_sizes: The shape of filter. 4-D with shape [batch, channels, height, weight].

    strides: A tuple/list of ints. The stride of the sliding window.

    pads: "SAME"or"VALID", indicating the type of pads algorithm to use, or tuple/list.

    dilations: An optional tuple/list of ints. Default to (1, 1, 1, 1).

    groups : The number of filter's group. Default value is 1.

    x_dtype: The dtype of feature map data. Default to float16.

    out_backprop_dtype: The dtype of gradients data. Default to float16.

    res_dtype: The dtype of result(De/Dw) data. Default to float32.

    kernel_name: Cce kernel name. Default to "conv2d_backprop_filter_cce".

    Returns : None
    ----------
    """

    def _ceil(x_1, x_2):
        if x_2 == 0:
            dict_args = {}
            dict_args["errCode"] = "E60108"
            dict_args["reason"] = "Division by zero"
            raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))
        return (x_1 + x_2 - 1) // x_2

    # dtype chek
    x_dtype = x_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    if x_dtype != out_backprop_dtype:
        dict_args = {}
        dict_args["errCode"] = "E60038"
        dict_args["desc"] = "The fmap data type is not same as the out_backprop data type."
        raise RuntimeError(dict_args, error_manager.get_error_message(dict_args))

    res = check_conv2dbp_filter_params(
        shape_x,
        shape_out_backprop,
        filter_sizes,
        strides,
        pads,
        dilations,
        groups,
        x_dtype,
        out_backprop_dtype,
        res_dtype,
        kernel_name,
    )
    (
        shape_x,
        shape_out_backprop,
        filter_sizes,
        strides,
        pads,
        dilations,
        groups,
        x_dtype,
        out_backprop_dtype,
        res_dtype,
        kernel_name,
    ) = res

    fmap_batch, fmap_channel, fmap_h, fmap_w = shape_x
    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop

    # x_dtype is same as outbackprop, use x_dtype to get C0 size
    c0_size = tbe_platform.CUBE_MKN.get(x_dtype).get("mac")[1]
    # Channel axis should be align with 16
    aligned_dedy_channel = _align(dedy_channel, tbe_platform.C0_SIZE)
    aligned_fmap_channel = _align(fmap_channel, tbe_platform.C0_SIZE)
    # C0 size is according to dtype, fp32 -> 8 and fp16-> 16
    shape_dedy = (dedy_batch, _ceil(aligned_dedy_channel, c0_size), dedy_h, dedy_w, c0_size)
    shape_fmap = (fmap_batch, _ceil(aligned_fmap_channel, c0_size), fmap_h, fmap_w, c0_size)
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    fmap = tvm.placeholder(shape_fmap, name="fmap", dtype=x_dtype)

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "groups": groups,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name
    }

    dedw = tbe.conv2d_backprop_filter(input_x=fmap,
                                      out_backprop=dedy,
                                      filter_sizes=filter_sizes,
                                      para_dict=para_dict)
    tensor_list_input = [fmap, dedy]

    with tvm.target.cce():
        sch = tbe.auto_schedule(dedw)

    real_outs = sch.cce_special["real_out_tensor"]
    tensor_list = tensor_list_input + real_outs
    config = {"name": kernel_name, "tensor_list": tensor_list}

    tbe.build(sch, config)


@tbe_platform.fusion_manager.register("conv2d_backprop_filter_d")
def conv2d_backprop_filter_compute(
    x,
    out_backprop,
    y,
    filter_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    kernel_name="conv2d_backprop_filter"
):
    def _calc_input_and_out_shape(origin_format_x,
                                  origin_shape_x,
                                  origin_format_res,
                                  origin_shape_res):
        if origin_format_x == "NHWC":
            x_shape = (origin_shape_x[0], origin_shape_x[3], origin_shape_x[1], origin_shape_x[2])
        elif origin_format_x == "NCHW":
            x_shape = origin_shape_x
        else:
            error_manager_cube.raise_err_input_format_invalid("Conv2dBackpropFilterD", "x", 
                                                              "[NCHW, NHWC]", origin_format_x)
        if origin_format_res == "NCHW":
            shape_res = origin_shape_res
        elif origin_format_res == "NHWC":
            shape_res = (
                origin_shape_res[0],
                origin_shape_res[3],
                origin_shape_res[1],
                origin_shape_res[2]
            )
        elif origin_format_res == "HWCN":
            shape_res = (
                origin_shape_res[3],
                origin_shape_res[2],
                origin_shape_res[0],
                origin_shape_res[1]
            )
        else:
            error_manager_cube.raise_err_input_format_invalid("Conv2dBackpropFilterD", "y", 
                                                            "[NCHW, NHWC, HWCN]", origin_format_res)
        return x_shape, shape_res

    origin_format_x = x.op.attrs["ori_format"]
    origin_shape_x = tuple(i.value for i in x.op.attrs["ori_shape"])
    res_dtype = y["dtype"]
    origin_shape_res = y["ori_shape"]
    origin_format_res = y["ori_format"]
    shape_x, shape_res = _calc_input_and_out_shape(origin_format_x,
                                                   origin_shape_x,
                                                   origin_format_res,
                                                   origin_shape_res)

    if len(strides) == 4:
        h_index = data_format.find("H")
        w_index = data_format.find("W")
        strides = [strides[h_index], strides[w_index]]
    if data_format == "NCHW":
        dilations = dilations
    elif data_format == "NHWC":
        dilations = (dilations[0], dilations[3], dilations[1], dilations[2])

    _, _, fmap_h, fmap_w = shape_x
    _, _, filter_h, filter_w = shape_res
    stride_h, stride_w = strides
    _, _, dilation_h, dilation_w = dilations
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    #pads calculate
    if pads == "SAME":
        pad_w = _align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = _align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pads = [pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = PADDING_VAILD
    pads = list(pads)

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "groups": groups,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name
    }

    dedw = tbe.conv2d_backprop_filter(input_x=x,
                                      out_backprop=out_backprop,
                                      filter_sizes=shape_res,
                                      para_dict=para_dict)
    return dedw