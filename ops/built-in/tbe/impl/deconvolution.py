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
deconvolution
"""
from impl.util import util_deconv_comm
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000

NoneType = type(None)

# memory type
MEMORY_DDR = 0
MEMORY_L1 = 1
MEMORY_L2 = 2
DECONV_SHAPE_DIM = 4

# 2 means L1 enable
L1FUSION_INPUT_CTR = 2

# the shape dim limited
INOUT_HW_MIN = 1
INOUT_H_MAX = 200000
INOUT_W_MAX = 4096
FILTER_HW_MIN = 1
STRIDE_HW_MIN = 1
DILATION_HW_MIN = 1
DILATION_HW_MAX = 255
CONV1D_W_MAX = 2147483647

# position index
N_DIM = 0
H_DIM = 2
W_DIM = 3


def _check_param(  # pylint: disable=invalid-name,R0913,R0914,W0613
    x,
    weight,
    y,
    strides,
    pads,
    dilations,
    data_format,
    offset_x):
    """
    the params check of deconvolution
    """
    ori_shape_x = x.get("ori_shape")
    ori_shape_filters = weight.get("ori_shape")
    ori_shape_res = y.get("ori_shape")
    x_dtype = x.get("dtype").lower()
    ori_format_x = x.get("ori_format")
    ori_format_filters = weight.get("ori_format")
    ori_format_res = y.get("ori_format")

    # only quantify with offset
    if x_dtype != "int8" and offset_x != 0:
        reason = "when x_dtype is fp16, offset_x must be 0"
        args_dict = {"errCode": "E60108", "reason": reason}
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))

    if len(strides) == DECONV_SHAPE_DIM:
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

    util_deconv_comm.check_attr_range("the h of filter", shape_filters[2], FILTER_HW_MIN)
    util_deconv_comm.check_attr_range("the w of filter", shape_filters[3], FILTER_HW_MIN)

    util_deconv_comm.check_attr_range("the h of stride", strides[0], STRIDE_HW_MIN)
    util_deconv_comm.check_attr_range("the w of stride", strides[1], STRIDE_HW_MIN)

    util_deconv_comm.check_attr_range("the h of dilations", dilations[2], DILATION_HW_MIN, DILATION_HW_MAX)
    util_deconv_comm.check_attr_range("the w of dilations", dilations[3], DILATION_HW_MIN, DILATION_HW_MAX)

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
    inout_h_max = INOUT_H_MAX
    inout_w_max = INOUT_W_MAX
    if fmap_h_pad == 1 and filter_h_dilation == 1 and strides[0] == 1:
        inout_w_max = CONV1D_W_MAX

    util_deconv_comm.check_attr_range("the h of fmap(output)", fmap_h, inout_limit_min, inout_h_max)
    util_deconv_comm.check_attr_range("the w of fmap(output)", fmap_w, inout_limit_min, inout_w_max)

    util_deconv_comm.check_attr_range("the h of dedy(input) after expands",
                                      dedy_h * strides[0], inout_limit_min, inout_h_max)

    if filter_h == 1 and filter_w == 1:
        util_deconv_comm.check_attr_range("the w of dedy after expands",
                          dedy_w * strides[0] * strides[1], inout_limit_min, inout_w_max)
    else:
        util_deconv_comm.check_attr_range("the w of dedy after expands",
                          dedy_w * strides[1], inout_limit_min, inout_w_max)


def check_supported(x,
                    weight,
                    bias,
                    offset_w,
                    y,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NCHW",
                    offset_x=0,
                    kernel_name="deconvolution"):
    """
    The h of x or y must be in [1, 200000].\n
    The w of x or y must be in [1, 4096].\n
    The h and w of weight or dilations must be in [1, 255].\n
    The h and w of strides must be in [1, 63].\n
    The n and c of dilations must be 1.\n
    The h and w must meet:\n
       hi - (hk - 1)*dk + 1 + padh // strideh = ho\n
       wi - (wk - 1)*wk + 1 + padw // stridew = wo\n
    """
    shape_x = x.get("ori_shape")
    dynamic_flag = any([i < 0 for i in shape_x])
    if dynamic_flag:
        return True, ""
    try:
        _check_param(x, weight, y, strides, pads, dilations, data_format, offset_x)
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
    if (list(pads)[::2] != [0, 0] or list(shape_filters[2:]) != [1, 1]) \
            and list(strides) != [1, 1]:
        al1_size = _cal_al1_size()
    return al1_size + bl1_size


def get_op_support_info(x,  # pylint: disable=invalid-name,R0913,R0914,W0613
                        weight,
                        bias,
                        offset_w,
                        y,
                        strides,
                        pads,
                        dilations=(1, 1, 1, 1),
                        groups=1,
                        data_format="NCHW",
                        offset_x=0,
                        kernel_name="deconvolution"):
    """
    get the deconvolution split
    """
    format_x = x.get("format")
    dtype_x = x.get("dtype")
    shape_filters = util_deconv_comm.get_filter_shape(weight.get("ori_format"), weight.get("ori_shape"))
    head_overlap_h = -1 if (shape_filters[2] == 1 and strides[0] == 1) else 0
    tail_overlap_h = head_overlap_h
    head_overlap_w = -1 if (shape_filters[3] == 1 and strides[1] == 1) else 0
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

    min_l1space = _cal_min_l1space(x, weight, y, strides, dilations, pads)
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
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.OPTION_ATTR_INT,
    para_check.KERNEL_NAME,
)
def deconvolution(  # pylint: disable=invalid-name,R0913,R0914,W0613
    x,
    weight,
    bias,
    offset_w,
    y,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NCHW",
    offset_x=0,
    kernel_name="deconvolution",
):
    """
    algorithm: deconvolution

    Parameters
    ----------
    x: dict with keys(shape and dtype)
                  The shape of gradients.

    weight: dict with keys(shape and dtype)
            input weight tensor

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: the offset for weight

    y: dict with keys(shape and dtype)
       deconvolution output tensor, dtype must be assigned

    strides: tuple/list of 2 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated deconvolution
    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    offset_x: offset of gradients in quant mode

    kernel_name: str
                 kernel name, default value is "deconvolution"

    Returns
    -------
    None
    """

    ori_shape_x = x.get("ori_shape")
    ori_shape_filters = weight.get("ori_shape")
    ori_shape_res = y.get("ori_shape")

    x_dtype = x.get("dtype")
    filters_dtype = weight.get("dtype")
    res_dtype = y.get("dtype")

    ori_format_x = x.get("ori_format")
    ori_format_filters = weight.get("ori_format")
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

    if len(strides) == DECONV_SHAPE_DIM:
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

    # get fusion para in L1 fusion
    fusion_para = _get_deconvolution_fusion_para(x, y)

    dilations = util_deconv_comm.get_shape_dilation(ori_format_x, dilations)

    bias_flag = bias is not None

    _deconvolution_cce(
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
        fusion_para=fusion_para,
        kernel_name=kernel_name,
        ori_format_weight=ori_format_filters
    )


def _get_deconvolution_fusion_para(input_x, input_y=None):
    """
    get fusion para for L1 fusion
    """
    input_memory_type = _get_value(input_x, "addr_type", 0)
    l1_fusion_type = _get_value(input_x, "L1_fusion_type", -1)
    fmap_l1_addr_flag = _get_value(input_x, "L1_addr_flag", False)
    fmap_l1_valid_size = _get_value(input_x, "L1_valid_size", 0)
    output_memory_type = (
        _get_value(input_y, "addr_type", 0) if input_y is not None else "fuse_flag"
    )
    l1_fusion_enable_flag = tbe_build.get_L1_info("L1_fusion_enabled")
    if input_memory_type not in (MEMORY_DDR, MEMORY_L1, MEMORY_L2):
        args_dict = {
            "errCode": "E65008",
            "input_memory_type_range": "(MEMORY_DDR, MEMORY_L1, MEMORY_L2)",
            "input_memory_type": str(input_memory_type),
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))

    if input_y is not None and output_memory_type not in (MEMORY_DDR, MEMORY_L1, MEMORY_L2):
        args_dict = {
            "errCode": "E65009",
            "output_memory_type_range": "(MEMORY_DDR, MEMORY_L1, MEMORY_L2)",
            "output_memory_type": str(output_memory_type),
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))

    if not l1_fusion_enable_flag:
        input_memory_type = 0
        if input_y is not None:
            output_memory_type = 0
        l1_fusion_type = -1
        fmap_l1_addr_flag = False
        fmap_l1_valid_size = 0
    fusion_para = {
        "input_memory_type": input_memory_type,
        "output_memory_type": output_memory_type,
        "l1_fusion_type": l1_fusion_type,
        "fmap_l1_addr_flag": fmap_l1_addr_flag,
        "fmap_l1_valid_size": fmap_l1_valid_size,
    }
    return fusion_para


@tbe_platform.fusion_manager.register("deconvolution")
def deconvolution_compute(  # pylint: disable=invalid-name,R0913,R0914,W0613
    x,
    weight,
    bias,
    offset_w,
    y,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NCHW",
    offset_x=0,
    kernel_name="deconvolution",
):
    """
    used for fusion
    Parameters
    ----------
    x: dict with keys(shape and dtype)
                  The shape of gradients.

    weight: dict with keys(shape and dtype)
            input weight tensor

    offset_w: the offset for weight

    bias: dict with keys(shape and dtype)
        The shape of bias.

    y: dict with keys(shape and dtype)
       deconvolution output tensor, dtype must be assigned

    strides: tuple/list of 2 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated deconvolution
    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    offset_x: offset of gradients in quant mode

    kernel_name: str
                 kernel name, default value is "deconvolution"

    Returns
    -------
    None
    """

    ori_shape_weight = [i.value for i in weight.op.attrs["ori_shape"]]
    ori_shape_x = [i.value for i in x.op.attrs["ori_shape"]]
    shape_x_5hd = [i.value for i in x.shape]
    ori_shape_res = y["ori_shape"]

    weight_dtype = weight.dtype
    x_dtype = x.dtype
    res_dtype = y["dtype"]

    ori_format_weight = weight.op.attrs["ori_format"]
    ori_format_x = x.op.attrs["ori_format"]
    ori_format_res = y["ori_format"]

    fusion_para = _get_deconvolution_fusion_para(x)
    if len(strides) == DECONV_SHAPE_DIM:
        h_index = data_format.find("H")
        w_index = data_format.find("W")
        strides = [strides[h_index], strides[w_index]]

    shape_weight = util_deconv_comm.get_filter_shape(
        ori_format_weight, ori_shape_weight
    )
    if weight_dtype == "int8":
        # NCHW means (groups * cout_ori, cin_ori, hk,wk), but it means
        # (groups * cin_ori, cout_ori, hk, wk) in int8 scenes.
        if shape_weight[0] % groups != 0:
            args_dict = {
                "errCode": "E60108",
                "reason": "batch of weight % groups must be 0",
            }
            raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
        shape_weight[0], shape_weight[1] = (
            shape_weight[1] * groups,
            shape_weight[0] // groups,
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
        shape_weight,
        groups,
        weight_dtype,
        ori_format_weight
    )

    util_deconv_comm.check_conv2dbp_input_params(
        shape_weight,
        shape_x,
        shape_res,
        strides,
        pads,
        dilations,
        weight_dtype,
        x_dtype,
        res_dtype,
        kernel_name,
        fusion_para,
        group_dict=group_dict
    )

    pads = util_deconv_comm.get_padlist(
        pads, shape_res, strides, shape_weight, dilations
    )

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "tensor_bias": bias,
        "offset_x": offset_x,
        "fusion_para": fusion_para,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "is_fusion_flag": True
    }

    res = tbe.conv2d_backprop_input(weight, x, shape_weight, shape_res, para_dict=para_dict)
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
    (dict, NoneType),
    str,
    str,
)
def _deconvolution_cce(  # pylint: disable=R0913, R0914
    shape_filter,
    shape_x,
    input_sizes,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    filter_dtype="float16",
    x_dtype="float16",
    res_dtype="float16",
    bias=False,
    offset_x=0,
    fusion_para=None,
    kernel_name="deconvolution_cce",
    ori_format_weight="NCHW",
):
    """
    Topi interface of deconvolution

    Parameters:
    ----------
    shape_filter : The shape of filter.
                   4-D with shape [batch, channels, height, width].

    shape_x : The shape of gradients.
              4-D with shape [batch, channels, height, width].

    input_sizes : The shape of feature map.
                  4-D with shape [batch, channels, height, width].

    strides : A list of ints. The stride of the sliding window.

    pads : "SAME"or"VALID" indicating the type of pads algorithm to use,
           or list.

    dilations : An optional tuple of ints. Default value is (1, 1, 1, 1).

    groups : The params of group_dict. Default to 1.

    filter_dtype : The dtype of filter data. Default value is float16.

    x_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    bias: False: no bias, True: have bias

    offset_x: offset of gradients in quant mode

    fusion_para: the L1 fusion para

    kernel_name : Cce kernel name. Default value is "deconvolution_cce"

    ori_format_weight : The original format of weight. Default to NCHW.

    Returns : None
    ----------
    """

    if filter_dtype == "int8" and x_dtype == "int8":
        # NCHW means (groups * cout_ori, cin_ori, hk,wk), but it means
        # (groups * cin_ori, cout_ori, hk, wk) in int8 scenes.
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
        input_sizes,
        shape_filter,
        groups,
        filter_dtype,
        ori_format_weight
    )
    res = util_deconv_comm.check_conv2dbp_input_params(
        shape_filter,
        shape_x,
        input_sizes,
        strides,
        pads,
        dilations,
        filter_dtype,
        x_dtype,
        res_dtype,
        kernel_name,
        fusion_para,
        group_dict=group_dict
    )

    (
        shape_filter,
        shape_x,
        input_sizes,
        strides,
        pads,
        dilations,
        filter_dtype,
        x_dtype,
        res_dtype,
        kernel_name,
    ) = res

    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_x
    _, _, filter_h, filter_w = shape_filter

    _, dy_k0, _ = tbe_platform.CUBE_MKN[x_dtype]["mac"]
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

    tensor_bias = None
    if bias:
        input_channel = util_deconv_comm.align(input_sizes[1], w_n0)
        tensor_bias = tvm.placeholder(
            (input_channel,), name="tensor_bias", dtype=res_dtype
        )

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "tensor_bias": tensor_bias,
        "offset_x": offset_x,
        "fusion_para": fusion_para,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    dedx = tbe.conv2d_backprop_input(filters=tensor_filter_frac,
                                     out_backprop=tensor_dedy,
                                     filter_sizes=shape_filter,
                                     input_sizes=input_sizes,
                                     para_dict=para_dict)

    if bias:
        tensor_list = [tensor_dedy, tensor_filter_frac, tensor_bias, dedx]
    else:
        tensor_list = [tensor_dedy, tensor_filter_frac, dedx]

    with tvm.target.cce():
        sch = tbe.auto_schedule(dedx)

    config = {"name": kernel_name, "tensor_list": tensor_list}

    tbe.build(sch, config)


def _get_value(obj, key, default=None):
    """
    get value from obj by key with default value
    obj supports type Tensor and dict
    """
    if isinstance(obj, tvm.tensor.Tensor):
        tensor_value = obj.op.attrs[key] if key in obj.op.attrs else default
        return tensor_value.value if hasattr(tensor_value, "value") else tensor_value
    return obj.get(key, default)
