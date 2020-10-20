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
import te.lang.cce as tbe
import te.platform as tbe_platform
from impl.util import util_deconv_comm
from te import tvm
from te.lang.cce.te_compute.cube_util import shape_to_list
from te.utils import error_manager
from te.utils import para_check

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

    shape_x = util_deconv_comm.get_shape_out_backprop(ori_format_x, ori_shape_x)

    shape_res = util_deconv_comm.get_shape_res(ori_format_res, ori_shape_res)

    # get fusion para in L1 fusion
    fusion_para = _get_deconvolution_fusion_para(x, y)
    valid_shape = fusion_para.get("valid_shape")
    if valid_shape and valid_shape[2] == shape_x[2]:
        fusion_para["valid_shape"] = ()
        fusion_para["slice_offset"] = ()

    dilations = util_deconv_comm.get_shape_dilation(ori_format_x, dilations)

    bias_flag = bias is not None

    _deconvolution_cce(
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
        fusion_para=fusion_para,
        kernel_name=kernel_name,
    )


def _get_deconvolution_fusion_para(input_x, input_y=None):
    """
    get fusion para for L1 fusion
    """
    input_memory_type = _get_value(input_x, "addr_type", 0)
    valid_shape = _get_value(input_x, "valid_shape", ())
    slice_offset = _get_value(input_x, "slice_offset", ())
    l1_fusion_type = _get_value(input_x, "L1_fusion_type", -1)
    fmap_l1_addr_flag = _get_value(input_x, "L1_addr_flag", False)
    fmap_l1_valid_size = _get_value(input_x, "L1_valid_size", 0)
    output_memory_type = (
        _get_value(input_y, "addr_type", 0) if input_y is not None else "fuse_flag"
    )
    l1_fusion_enable_flag = tbe_platform.get_L1_info("L1_fusion_enabled")
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
    if valid_shape and not slice_offset:
        reason = "valid shape exists, slice shape cannot be []"
        args_dict = {"errCode": "E60108", "reason": reason}
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))

    valid_shape = shape_to_list(valid_shape)
    slice_offset = shape_to_list(slice_offset)
    if not l1_fusion_enable_flag:
        input_memory_type = 0
        if input_y is not None:
            output_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1
        fmap_l1_addr_flag = False
        fmap_l1_valid_size = 0
    fusion_para = {
        "input_memory_type": input_memory_type,
        "output_memory_type": output_memory_type,
        "valid_shape": valid_shape,
        "slice_offset": slice_offset,
        "l1_fusion_type": l1_fusion_type,
        "fmap_l1_addr_flag": fmap_l1_addr_flag,
        "fmap_l1_valid_size": fmap_l1_valid_size,
    }
    return fusion_para


@tbe_platform.fusion_manager.fusion_manager.register("deconvolution")
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

    if weight_dtype == "int8":
        ori_shape_weight[0], ori_shape_weight[1] = (
            ori_shape_weight[1],
            ori_shape_weight[0],
        )
    shape_weight = util_deconv_comm.get_filter_shape(
        ori_format_weight, ori_shape_weight
    )
    shape_x = util_deconv_comm.get_shape_out_backprop(ori_format_x, ori_shape_x)
    shape_res = util_deconv_comm.get_shape_res(ori_format_res, ori_shape_res)
    dilations = util_deconv_comm.get_shape_dilation(ori_format_x, dilations)

    valid_shape = fusion_para.get("valid_shape")
    if valid_shape and valid_shape[2] == shape_x[2]:
        fusion_para["valid_shape"] = ()
        fusion_para["slice_offset"] = ()

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
    )

    pads = util_deconv_comm.get_padlist(
        pads, shape_res, strides, shape_weight, dilations
    )

    res = tbe.conv2d_backprop_input_compute(
        weight,
        x,
        shape_weight,
        shape_res,
        strides,
        pads,
        dilations,
        res_dtype=res_dtype,
        tensor_bias=bias,
        offset_x=offset_x,
        fusion_para=fusion_para,
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
    (dict, NoneType),
    str,
)
def _deconvolution_cce(  # pylint: disable=R0913, R0914
    shape_filter,
    shape_x,
    input_sizes,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    filter_dtype="float16",
    x_dtype="float16",
    res_dtype="float16",
    bias=False,
    offset_x=0,
    fusion_para=None,
    kernel_name="deconvolution_cce",
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

    filter_dtype : The dtype of filter data. Default value is float16.

    x_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    bias: False: no bias, True: have bias

    offset_x: offset of gradients in quant mode

    fusion_para: the L1 fusion para

    kernel_name : Cce kernel name. Default value is "deconvolution_cce"

    Returns : None
    ----------
    """


    if not fusion_para:
        fusion_para = {
            "input_memory_type": 0,
            "output_memory_type": 0,
            "valid_shape": (),
            "slice_offset": (),
            "l1_fusion_type": -1,
            "fmap_l1_addr_flag": False,
            "fmap_l1_valid_size": 0,
        }

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
        input_sizes,
        strides,
        pads,
        dilations,
        filter_dtype,
        x_dtype,
        res_dtype,
        kernel_name,
        fusion_para,
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
    filter_batch, filter_channel, filter_h, filter_w = shape_filter

    _, dy_k0, _ = tbe_platform.CUBE_MKN[x_dtype]["mac"]
    _, w_k0, w_n0 = tbe_platform.CUBE_MKN[filter_dtype]["mac"]
    shape_dedy = (
        dedy_batch,
        util_deconv_comm.ceil(dedy_channel, dy_k0),
        dedy_h,
        dedy_w,
        dy_k0,
    )
    filter_channel = util_deconv_comm.align(filter_channel, w_n0)
    if filter_dtype == "int8" and x_dtype == "int8":
        shape_filter_frac = (
            util_deconv_comm.ceil(filter_batch, w_k0) * filter_h * filter_w,
            util_deconv_comm.ceil(filter_channel, w_n0),
            w_n0,
            w_k0,
        )
    else:
        shape_filter_frac = (
            util_deconv_comm.ceil(filter_channel, w_n0) * filter_h * filter_w,
            util_deconv_comm.ceil(filter_batch, w_k0),
            w_k0,
            w_n0,
        )
    tensor_dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=x_dtype)

    tensor_filter_frac = tvm.placeholder(
        shape_filter_frac, name="filter", dtype=filter_dtype
    )

    tensor_bias = None
    if bias:
        tensor_bias = tvm.placeholder(
            (filter_channel,), name="tensor_bias", dtype=res_dtype
        )

    dedx = tbe.conv2d_backprop_input_compute(
        filters=tensor_filter_frac,
        out_backprop=tensor_dedy,
        filter_sizes=shape_filter,
        input_sizes=input_sizes,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype,
        tensor_bias=tensor_bias,
        offset_x=offset_x,
        fusion_para=fusion_para,
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


def _get_value(obj, key, default=None):
    """
    get value from obj by key with default value
    obj supports type Tensor and dict
    """
    if isinstance(obj, tvm.tensor.Tensor):
        tensor_value = obj.op.attrs[key] if key in obj.op.attrs else default
        return tensor_value.value if hasattr(tensor_value, "value") else tensor_value
    return obj.get(key, default)
