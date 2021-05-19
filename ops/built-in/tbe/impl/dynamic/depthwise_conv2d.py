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
depthwise_conv2d
"""
from tbe import tvm
from tbe.common import register as tbe_register
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import error_manager_cube as err_man
from .conv2d import conv2d
from .conv2d import conv2d_fusion_compute

NONETYPE = type(None)

@tbe_register.register_op_compute("DepthwiseConv2D", op_mode="dynamic", support_fusion=True)
@para_check.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (tvm.tensor.Tensor, NONETYPE),
                             (tvm.tensor.Tensor, NONETYPE), dict, (tuple, list), (tuple, list),
                             (tuple, list), str, int, str)
def depthwise_compute(fmap,
                      filter,
                      bias,
                      offset_w,
                      out,
                      strides,
                      dilations,
                      pads,
                      data_format='NHWC',
                      offset_x=0,
                      kernel_name="depthwise_conv2d"):
    """
    algorithm: depthwise conv2d compute
    calculating  depthwise compute
    Parameters
    ----------
    fmap : a tensor of featureMap
    filter : a tensor of filter
    bias : a tensor of bias
    offset_w : a tensor of filter offset
    out : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.
    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]
    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]
    pads : padding added to each dimension of the input
    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]
    offset_x : offset of the input
    Returns
    -------
    None
    """
    filter_format = filter.op.attrs['ori_format'].value
    if filter_format == "HWCN":
        groups = filter.op.attrs['ori_shape'][3].value
    elif filter_format == "NCHW" or filter_format == "NHWC":
        groups = filter.op.attrs['ori_shape'][0].value
    else:
        err_man.raise_err_input_format_invalid("depthwise_conv2d", "filter", \
            ["HWCN", "NCHW", "NHWC"], filter["ori_format"])

    out = conv2d_fusion_compute(fmap, filter, bias, offset_w, out, strides, pads, dilations,
                                groups=groups, data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)
    return out


@tbe_register.register_operator("DepthwiseConv2D")
@para_check.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list),
                             str, int, str)
def depthwise_conv2d(
        x,
        filter,
        bias,
        offset_w,
        y,
        strides,
        dilations=(1, 1, 1, 1),
        pads=(0, 0, 0, 0),
        data_format='NHWC',
        offset_x=0,
        kernel_name="depthwise_conv2d",
):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution

    Parameters
    ----------
    x : a dict of featureMap
        {"shape", "dtype", "format"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    filter : a dict of filter
        {"shape", "dtype"}
        shape of filter tensor [C1, H, W, K, Co, C0],
        K is depthwise_multiplier, support int.

    bias : a dict of bias
        {"shape", "dtype"}
        shape of bias tensor [C1*C0,]
        support int8.

    offset_w : a dict of filter offset
        {"shape", "dtype"}
        shape of offset tensor [C1, H, W, K, Co, C0]
        support float16.

    y : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]

    pads : padding added to each dimension of the input

    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    offset_x : offset of the input

    kernel_name : str
       cce kernel name

    Returns
    -------
    None

    """
    x_shape = x.get("ori_shape")
    if x["ori_format"] == "NCHW":
        x_c = x_shape[1]
    elif x["ori_format"] == "NHWC":
        x_c = x_shape[3]
    else:
        err_man.raise_err_input_format_invalid("depthwise_conv2d", "x", \
            ["NCHW", "NHWC"], x["ori_format"])

    w_shape = filter.get("ori_shape")
    if filter["ori_format"] == "HWCN":
        filter_n = w_shape[3] * w_shape[2]
        filter_c = w_shape[2]
        filter_h = w_shape[0]
        filter_w = w_shape[1]
    elif filter["ori_format"] == "NCHW":
        filter_n = w_shape[0] * w_shape[1]
        filter_c = w_shape[1]
        filter_h = w_shape[2]
        filter_w = w_shape[3]
    elif filter["ori_format"] == "NHWC":
        filter_n = w_shape[0] * w_shape[3]
        filter_c = w_shape[3]
        filter_h = w_shape[1]
        filter_w = w_shape[2]
    else:
        err_man.raise_err_input_format_invalid("depthwise_conv2d", "filter", \
            ["HWCN", "NCHW", "NHWC"], filter["ori_format"])

    filter["ori_shape"] = [filter_n, 1, filter_h, filter_w]
    filter["ori_format"] = "NCHW"

    conv2d(x, filter, bias, offset_w, y, strides, pads, dilations,
           groups=x_c, data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)

