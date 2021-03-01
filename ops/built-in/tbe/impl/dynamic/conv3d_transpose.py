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
conv3d_transpose
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
import te.lang.base as tbe_base
from tbe.dsl.compute import conv3d_backprop_input_compute as conv3d_bp_dx
from impl.dynamic import check_and_config_para
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.errormgr import error_manager_cube as cube_err
from te import tvm


Nonetype = type(None)


def _check_output_padding(output_padding, stride, dilation, data_format):
    def _ncdhw2ndhwc(shape_ncdhw):
        shape_ndhwc = [shape_ncdhw[0], shape_ncdhw[2], shape_ncdhw[3], shape_ncdhw[4], shape_ncdhw[1]]
        return shape_ndhwc
    if data_format == "NCDHW":
        output_padding = _ncdhw2ndhwc(output_padding)
        stride = _ncdhw2ndhwc(stride)
        dilation = _ncdhw2ndhwc(dilation)
    _, output_padding_d, output_padding_h, output_padding_w, _ = output_padding
    _, stride_d, stride_h, stride_w, _ = stride
    _, dilation_d, dilation_h, dilation_w, _ = dilation
    if output_padding_d < 0 or (output_padding_d >= dilation_d and output_padding_d >= stride_d):
        cube_err.raise_err_three_paras(
            'E62305', 'conv3d_transpose', 'output_padding D', 
            '[{}, {})'.format(str(0), 'max(stride D,dilation D)'), str(output_padding_d))

    if output_padding_h < 0 or (output_padding_h >= dilation_h and output_padding_h >= stride_h):
        cube_err.raise_err_three_paras(
            'E62305', 'conv3d_transpose', 'output_padding H', 
            '[{}, {})'.format(str(0), 'max(stride H,dilation H)'), str(output_padding_h))

    if output_padding_w < 0 or (output_padding_w >= dilation_w and output_padding_w >= stride_w):
        cube_err.raise_err_three_paras(
            'E62305', 'conv3d_transpose', 'output_padding W', 
            '[{}, {})'.format(str(0), 'max(stride W,dilation W)'), str(output_padding_w))


def _conv3d_transpose_compute(filters, out_backprop, y_input, input_size, strides, pads,
                              dilations=(1, 1, 1, 1, 1), groups=1, output_padding=(0,0,0,0,0),
                              data_format="NDHWC", kernel_name="conv3d_transpose"):
    _check_output_padding(output_padding, strides, dilations, data_format)
    res = check_and_config_para(
        filters, out_backprop, y_input, input_size, strides, pads, dilations, groups, data_format, kernel_name)
    (dx_shape, dedy, filter_frac, shape_filter, shape_out_backprop, input_sizes, strides,
     pads, dilations, filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict) = res

    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    shape_filter_ncdhw = [filter_batch, filter_channel, filter_depth, filter_h, filter_w]

    para_dict = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    dedx = conv3d_bp_dx.conv3d_dx(
        filter=filter_frac,
        out_backprop=dedy,
        filter_size=shape_filter_ncdhw,
        input_size=input_sizes,
        para_dict=para_dict
    )

    return {'op_placeholder': [dx_shape, dedy, filter_frac],  'op_res': [dedx]}

@tbe_base.register_operator("Conv3DTranspose")
@para_check.check_input_type(dict, dict, dict, (Nonetype, dict), (Nonetype, dict),
                             dict, (tuple, list), (tuple, list, str),
                             (tuple, list), int, str, (tuple, list), int, str)
def conv3d_transpose(input_size, x, filter, # pylint: disable=R0913,R0914
                     bias, offset_w, y, strides,
                     pads, dilations=(1, 1, 1, 1, 1), groups=1,
                     data_format="NDHWC", output_padding=(0, 0, 0, 0, 0),
                     offset_x=0, kernel_name="conv3d_transpose"):
    """
    algorithm: Conv3d_transpose

    Parameters
    ----------
    input_size: dict, will not be used
            input tensor size.

    x: A dict with keys(shape and dtype)
        Gradients tensor

    filter: A dict with keys(shape and dtype)
        Input weight tensor

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
        Input offset_w tensor.

    y: A dict with keys(shape and dtype)
        conv3d_transpose output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers
        Filter move stride

    pads: A tuple/list of 6 integers: [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
          str: "SAME" or "VALID"

    dilations: A tuple/list of 5 integers
        filter expand size of dilated conv3d_transpose, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
        Default value 1

    data_format: The data format of the input and output data
        Default format "NDHWC"

    output_padding: tuple/list of 5 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0, 0)

    offset_x: int
        offset of gradients in quant mode. Default to 0

    kernel_name: Str
        Kernel name, default value is "conv3d_transpose"

    Returns
    -------
    None
    """
    with tbe_base.compute():
        res = _conv3d_transpose_compute(filter, x, y, input_size, strides, pads,
                                        dilations, groups, output_padding,
                                        data_format, kernel_name=kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get("op_res"))

    tensor_list = res.get("op_placeholder") + res.get("op_res")
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False}
    }
    tbe.build(sch, config)
