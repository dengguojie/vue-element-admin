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
import te.lang.cce as tbe
import te.platform as tbe_platform
from impl.util import util_deconv_comm
from te import tvm
from te.utils import error_manager
from te.utils import para_check

# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT,
    para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_INT,
    para_check.REQUIRED_ATTR_STR,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_INT,
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
    groups=None,
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
            input filter tensor

    offset_w: the offset for filter

    bias: dict with keys(shape and dtype)
        The shape of bias.

    y: dict with keys(shape and dtype)
       conv2d_transpose_d output tensor, dtype must be assigned

    input_size: The shape of feature map.
                 4-D with shape [batch, channels, height, filter].

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_transpose_d
    groups: int
            param for group conv2d_transpose_d

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "conv2d_transpose_d"

    output_padding: The size will be added in the output shape.

    offset_x: the offset for x

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

    shape_x = util_deconv_comm.get_shape_out_backprop(ori_format_x, ori_shape_x)

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
        filter_dtype=filters_dtype,
        x_dtype=x_dtype,
        res_dtype=res_dtype,
        bias=bias_flag,
        offset_x=offset_x,
        kernel_name=kernel_name,
    )


@tbe_platform.fusion_manager.fusion_manager.register("conv2d_transpose_d")
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
    groups=None,
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

    offset_w: the offset for filter

    bias: dict with keys(shape and dtype)
        The shape of bias.

    y: dict with keys(shape and dtype)
       conv2d_transpose_d output tensor, dtype must be assigned

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_transpose_d
    groups: int
            param for group conv2d_transpose_d

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "conv2d_transpose_d"

    output_padding: The size will be added in the output shape.

    offset_x: the offset for x

    Returns
    -------
    None
    """
    ori_shape_filter = [i.value for i in filter.op.attrs["ori_shape"]]
    ori_shape_x = [i.value for i in x.op.attrs["ori_shape"]]
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
            ori_format_filter, ori_shape_filter
        )

    shape_filter = util_deconv_comm.get_filter_shape(
        ori_format_filter, ori_shape_filter
    )
    shape_x = util_deconv_comm.get_shape_out_backprop(ori_format_x, ori_shape_x)
    shape_res = util_deconv_comm.get_shape_res(ori_format_res, ori_shape_res)
    dilations = util_deconv_comm.get_shape_dilation(ori_format_x, dilations)

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
    )

    pads = util_deconv_comm.get_padlist(
        pads, shape_res, strides, shape_filter, dilations
    )

    res = tbe.conv2d_backprop_input_compute(
        filter,
        x,
        shape_filter,
        shape_res,
        strides,
        pads,
        dilations,
        res_dtype=res_dtype,
        tensor_bias=bias,
        offset_x=offset_x,
        kernel_name=kernel_name,
    )

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
    bool,
    int,
    str,
)
def _conv2d_transpose_cce(
    shape_filter,  # pylint: disable=R0913, R0914
    shape_x,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    filter_dtype="float16",
    x_dtype="float16",
    res_dtype="float16",
    bias=False,
    offset_x=0,
    kernel_name="conv2d_transpose_cce",
):
    """
    Topi interface of conv2d_transpose_d

    Parameters:
    ----------
    shape_filter : The shape of filter.
                   4-D with shape [batch, channels, height, width].

    shape_x : The shape of gradients.
              4-D with shape [batch, channels, height, width].

    input_size : The shape of feature map.
                  4-D with shape [batch, channels, height, width].

    strides : A list of ints. The stride of the sliding window.

    pads : "SAME"or"VALID" indicating the type of pads algorithm to use,
           or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    filter_dtype : The dtype of filter data. Default value is float16.

    x_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    bias: False: no bias, True: have bias

    kernel_name : Cce kernel name. Default value is "conv2d_transpose_cce"

    Returns : None
    ----------
    """

    def _ceil(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2

    if filter_dtype == "int8" and x_dtype == "int8":
        shape_filter = [
            shape_filter[1],
            shape_filter[0],
            shape_filter[2],
            shape_filter[3],
        ]
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
    filter_channel = util_deconv_comm.align(filter_channel, w_n0)
    if filter_dtype == "int8" and x_dtype == "int8":
        shape_filter_frac = (
            _ceil(filter_batch, w_k0) * filter_h * filter_w,
            _ceil(filter_channel, w_n0),
            w_n0,
            w_k0,
        )
    else:
        shape_filter_frac = (
            _ceil(filter_channel, w_n0) * filter_h * filter_w,
            _ceil(filter_batch, w_k0),
            w_k0,
            w_n0,
        )
    tensor_dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=x_dtype)

    tensor_filter_frac = tvm.placeholder(
        shape_filter_frac, name="filter", dtype=filter_dtype
    )

    if bias:
        tensor_bias = tvm.placeholder(
            (filter_channel,), name="tensor_bias", dtype=res_dtype
        )
    else:
        tensor_bias = None

    dedx = tbe.conv2d_backprop_input_compute(
        filters=tensor_filter_frac,
        out_backprop=tensor_dedy,
        filter_sizes=shape_filter,
        input_sizes=input_size,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype,
        tensor_bias=tensor_bias,
        offset_x=offset_x,
        kernel_name=kernel_name,
    )
    if bias:
        tensor_list = [tensor_dedy, tensor_filter_frac, tensor_bias, dedx]
    else:
        tensor_list = [tensor_dedy, tensor_filter_frac, dedx]

    with tvm.target.cce():
        sch = tbe.auto_schedule(dedx)

    config = {"name": kernel_name, "tensor_list": tensor_list}

    tbe.cce_build_code(sch, config)
