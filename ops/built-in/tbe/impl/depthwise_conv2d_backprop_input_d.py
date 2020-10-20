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
Depthwise conv2D backprop input for the computation of
gradients of depthwise convolution with respect to the input.
"""
import te.platform as tbe_platform
from te.utils import check_para
from te.utils.error_manager import error_manager_util
from te.utils.error_manager import error_manager_conv2d
from te.lang.cce.te_compute.depthwise_conv2d_compute import \
    depthwise_conv2d_backprop_input_d_compute
from te.lang.cce.te_schedule.depthwise_conv2d_schedule import \
    depthwise_conv2d_backprop_input_d_schedule

from te import tvm


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=too-many-lines
BLOCK_SIZE = tbe_platform.BLOCK_REDUCE

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 4

# shape's dim of filter must be 4
FILTER_DIM = 4

# shape's dim of strides must be 4
STRIDES_DIM = 4

# shape's dim of dilations must be 4
DILATIONS_DIM = 4


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=redefined-builtin
def _check_params(shape, dtype, format):
    """
    check the parameters including shape, dtype, format

    Parameters
    ----------
    shape : shape of tensor

    dtype : data type

    format : tensor format

    Returns
    -------
    None
    """
    if format == "NCHW":
        check_para.check_shape(shape,
                               min_rank=FEATURE_MAP_DIM,
                               max_rank=FEATURE_MAP_DIM)
    if format in ("HWCK", "HWCN"):
        check_para.check_shape(shape, min_rank=FILTER_DIM, max_rank=FILTER_DIM)
    check_para.check_dtype(dtype.lower(), ('float16', ))


# pylint: disable=locally-disabled, too-many-locals, too-many-statements
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=too-many-branches, redefined-builtin
@check_para.check_op_params(
    check_para.REQUIRED_INPUT, check_para.REQUIRED_INPUT,
    check_para.REQUIRED_OUTPUT,
    check_para.REQUIRED_ATTR_LIST_INT, check_para.REQUIRED_ATTR_LIST_INT,
    check_para.OPTION_ATTR_LIST_INT, check_para.REQUIRED_ATTR_LIST_INT,
    check_para.OPTION_ATTR_STR, check_para.KERNEL_NAME)
def depthwise_conv2d_backprop_input_d(
        filter,
        out_backprop,
        input_grad,
        input_size,
        strides,
        dilations=(1, 1, 1, 1),
        pads=(0, 0, 0, 0),
        data_format='NHWC',
        kernel_name="depthwise_conv2d_backprop_input"):
    """
    algorithm: depthwise conv2d backprop input

    computes the gradients of depthwise convolution with respect to the input

    Parameters
    ----------
    filter: dict
        4-D origin shape and dtype of filter tensor
        support [H, W, C, K], K is channel_multiplier

    out_backprop: dict
        4-D origin shape and dtype of out_backprop tensor,
        support [N, Co, Ho, Wo] or [N, Ho, Wo, Co],
        gradients w.r.t. the output of the convolution

    input_grad: dict
        4-D origin shape and dtype of input tensor,
        support [N, C, H, W] or [N, H, W, C]

    input_size: a list or tuple of four ints
        shape of input tensor, support [N, C, H, W] or [N, H, W, C]

    strides: a list or tuple of four ints
        the stride of the sliding window for height and width of the input of
        the convolution, support [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations: an optional list or tuple of four ints
        the dilation factor for each dimension of input
        if set to k > 1, there will be k-1 skipped cells between each
        filter element on that dimension, support [1, 1, dilation_height,
        dilation_width] or [1, dilation_height, dilation_width, 1]

    pads: a list or tuple of four ints
        padding added to each dimension of the input

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    kernel_name: str
        cce kernel name, default value is "depthwise_conv2d_backprop_input"

    Returns
    -------
    None
    """
    def _ceil(x):
        """
        Return the least multiple of 16 integer number
        which is greater than or equal to x.
        """
        return ((x + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    input_shape = input_grad.get("ori_shape")
    if input_size != input_shape:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'attr_name': 'shape',
            'param1_name': 'input_size',
            'param2_name': 'input_shape',
            'param1_value': str(input_size),
            'param2_value': str(input_shape)
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))
    input_dtype = input_grad.get("dtype").lower()
    filter_shape = filter.get("ori_shape")
    filter_dtype = filter.get("dtype").lower()

    output_shape = out_backprop.get("ori_shape")
    output_dtype = out_backprop.get("dtype").lower()

    input_ori_format = input_grad.get('ori_format')
    if input_ori_format != 'NCHW' and input_ori_format != 'NHWC':
        dict_args = {
            'errCode': 'E50002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param': 'original featuremap',
            'expected_format_list': '[{}, {}]'.format('NCHW', 'NHWC'),
            'format': input_ori_format
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))
    filter_ori_format = filter.get('ori_format')
    if filter_ori_format not in ('HWCK', 'HWCN', 'NCHW'):
        dict_args = {
            'errCode': 'E50002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param': 'original featuremap',
            'expected_format_list': '[{}, {}, {}]'.format(
                'HWCK', 'HWCN', 'NCHW'),
            'format': filter_ori_format
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))
    dout_ori_format = out_backprop.get('ori_format')
    if dout_ori_format != 'NCHW' and dout_ori_format != 'NHWC':
        dict_args = {
            'errCode': 'E50002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param': 'dout_ori_format',
            'expected_format_list': '[{}, {}]'.format('NCHW', 'NHWC'),
            'format': dout_ori_format
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))

    # index of the strides dimension
    DIM_S_N, DIM_S_C, DIM_S_H, DIM_S_W = 0, 1, 2, 3
    # index of the dilations dimension
    DIM_D_N, DIM_D_C, DIM_D_H, DIM_D_W = 0, 1, 2, 3
    # index of the out_backprop dimension
    DIM_N, DIM_C, _, _ = 0, 1, 2, 3
    # index of the filter dimension
    _, _, DIM_W_C, DIM_W_K = 0, 1, 2, 3

    if input_ori_format == 'NHWC':
        DIM_S_N, DIM_S_H, DIM_S_W, DIM_S_C = 0, 1, 2, 3
        DIM_D_N, DIM_D_H, DIM_D_W, DIM_D_C = 0, 1, 2, 3
        input_shape = [
            input_shape[0], input_shape[3], input_shape[1], input_shape[2]
        ]
    if dout_ori_format == 'NHWC':
        output_shape = [
            output_shape[0], output_shape[3], output_shape[1], output_shape[2]
        ]
    if filter_ori_format == "NCHW":
        filter_shape = [
            filter_shape[2], filter_shape[3], filter_shape[1], filter_shape[0]
        ]
    if data_format not in ('NCHW', 'NHWC'):
        dict_args = {
            'errCode': 'E50002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param': 'dout_ori_format',
            'expected_format_list': '[{}, {}]'.format('NCHW', 'NHWC'),
            'format': dout_ori_format
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))

    # check if the parameter is valid
    _check_params(filter_shape, filter_dtype, "HWCK")
    _check_params(output_shape, output_dtype, "NCHW")
    _check_params(input_shape, input_dtype, "NCHW")
    check_para.check_shape(output_shape,
                           min_rank=FEATURE_MAP_DIM,
                           max_rank=FEATURE_MAP_DIM,
                           param_name="out_backprop")
    check_para.check_shape(filter_shape,
                           min_rank=FILTER_DIM,
                           max_rank=FILTER_DIM,
                           param_name="filter")
    check_para.check_shape(input_shape,
                           min_rank=FEATURE_MAP_DIM,
                           max_rank=FEATURE_MAP_DIM,
                           param_name="input_grad")
    check_para.check_shape(strides,
                           min_rank=STRIDES_DIM,
                           max_rank=STRIDES_DIM,
                           param_name="strides")
    check_para.check_shape(dilations,
                           min_rank=DILATIONS_DIM,
                           max_rank=DILATIONS_DIM,
                           param_name="dilations")

    if strides[DIM_S_H] != strides[DIM_S_W]:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'attr_name': 'strides value',
            'param1_name': 'strides[DIM_S_H]',
            'param2_name': 'strides[DIM_S_W]',
            'param1_value': str(strides[DIM_S_H]),
            'param2_value': str(strides[DIM_S_W])
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))

    if (strides[DIM_S_N] != 1) or (strides[DIM_S_C] != 1):
        error_manager_conv2d.raise_err_specific_user(
            "depthwise_backprob_input",
            "the N-dim and C-dim of stride must be equal to 1.")

    if (dilations[DIM_D_N] != 1) or (dilations[DIM_D_C] != 1):
        dict_args = {
            'errCode': 'E62510',
            'op_name': 'depthwise_conv2d_backprop_input',
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))
    if input_shape[DIM_N] != output_shape[DIM_N]:
        dict_args = {
            'errCode': 'E64002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param1': 'input_shape[DIM_N]',
            'param2': 'output_shape[DIM_N]',
            'actual_value': str(input_shape[DIM_N])
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))
    if filter_shape[DIM_W_K] != 1:
        dict_args = {
            'errCode': 'E64002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param1': 'filter_shape[DIM_W_K]',
            'param2': '1',
            'actual_value': str(filter_shape[DIM_W_K])
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))
    if input_shape[DIM_C] != output_shape[DIM_C]:
        dict_args = {
            'errCode': 'E64002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param1': 'input_shape[DIM_C]',
            'param2': 'output_shape[DIM_C]',
            'actual_value': str(input_shape[DIM_C])
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))

    if (_ceil(input_shape[DIM_C]) //
            BLOCK_SIZE) != (_ceil(filter_shape[DIM_W_C]) // BLOCK_SIZE):

        dict_args = {
            'errCode': 'E64002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param1': '_ceil(input_shape[DIM_C])',
            'param2': '(_ceil(filter_shape[DIM_W_C]) // BLOCK_SIZE)',
            'actual_value': str(_ceil(input_shape[DIM_C]))
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))

    # check pad parameter
    if len(pads) != 4:
        dict_args = {
            'errCode': 'E50001',
            'param': 'pads',
            'op_name': 'depthwise_conv2d_backprop_input',
            'expected_length': "4",
            'length': str(len(pads))
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))

    # input parameters
    batch, input_channel, input_height, input_width = input_shape
    filter_height, filter_width, filter_channel, _ = filter_shape
    input_c1 = (input_channel + BLOCK_SIZE - 1) // BLOCK_SIZE
    stride_h, stride_w = strides[DIM_S_H], strides[DIM_S_W]
    dilation_h, dilation_w = dilations[DIM_D_H], dilations[DIM_D_W]
    strides = (stride_h, stride_w)
    dilations = (dilation_h, dilation_w)

    # output parameters
    batch, output_channel, output_height, output_width = output_shape
    output_c1 = (output_channel + BLOCK_SIZE - 1) // BLOCK_SIZE

    dilated_filter_height = (filter_height - 1) * dilation_h + 1
    dilated_filter_width = (filter_width - 1) * dilation_w + 1

    pad_top, pad_bottom, pad_left, pad_right = pads
    full_height = input_height + pad_top + pad_bottom
    full_width = input_width + pad_left + pad_right
    out_backprop_height = (full_height - dilated_filter_height) // stride_h + 1
    out_backprop_width = (full_width - dilated_filter_width) // stride_w + 1

    if output_height != out_backprop_height:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'attr_name': 'height value',
            'param1_name': 'output_height',
            'param2_name': 'out_backprop_height',
            'param1_value': str(output_height),
            'param2_value': str(out_backprop_height)
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))
    if output_width != out_backprop_width:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'attr_name': 'width value',
            'param1_name': 'output_width',
            'param2_name': 'out_backprop_width',
            'param1_value': str(output_width),
            'param2_value': str(out_backprop_width)
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))

    filter_shape = [
        _ceil(filter_channel) // BLOCK_SIZE, filter_height * filter_width,
        1, BLOCK_SIZE, BLOCK_SIZE
    ]
    filter_init = tvm.placeholder(filter_shape,
                                  dtype=filter_dtype,
                                  name='filter')

    output_shape = [
        batch, output_c1, 1, output_height, output_width, BLOCK_SIZE
    ]
    dout = tvm.placeholder(output_shape, dtype=output_dtype, name='dout')

    input_shape = [
        batch, input_c1, 1, input_height, input_width, BLOCK_SIZE
    ]
    res = depthwise_conv2d_backprop_input_d_compute(
        input_shape, filter_init, dout, [filter_height, filter_width],
        strides, pads, kernel_name)

    sch = depthwise_conv2d_backprop_input_d_schedule(res)
    with tbe_platform.build_config:
        tvm.build_module.build(
            sch, [filter_init, dout, res], "cce", name=kernel_name)
