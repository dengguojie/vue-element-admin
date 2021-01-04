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
from te.utils import para_check
from te.utils.error_manager import error_manager_util
from te.utils.error_manager import error_manager_cube
from te.lang.cce.te_compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_input_d_compute
from te.lang.cce.te_schedule.depthwise_conv2d_schedule import depthwise_conv2d_backprop_input_d_schedule
from te import tvm

BLOCK_SIZE = tbe_platform.BLOCK_REDUCE

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 4

# shape's dim of filter must be 4
FILTER_DIM = 4

# shape's dim of strides must be 4
STRIDES_DIM = 4

# shape's dim of dilations must be 4
DILATIONS_DIM = 4


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
        para_check.check_shape(shape, min_rank=FEATURE_MAP_DIM, max_rank=FEATURE_MAP_DIM)
    if format in ("HWCK", "HWCN"):
        para_check.check_shape(shape, min_rank=FILTER_DIM, max_rank=FILTER_DIM)
    para_check.check_dtype(dtype.lower(), ('float16', ))


def _ceil(x_size):
    """
    Return the least multiple of 16 integer number
    which is greater than or equal to x.
    """
    if BLOCK_SIZE:
        return ((x_size + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    dict_args = {'errCode': 'E67006', 'op_name': 'depthwise_conv2d_backprop_filter', 'param_name': 'param_name'}
    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_input_size(input_size, input_shape):
    """
    check input size
    """
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
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_data_format(data_format, expect_format_list):
    """
    check data format
    """
    if data_format not in expect_format_list:
        dict_args = {
            'errCode': 'E50002',
            'op_name': 'depthwise_conv2d',
            'param': 'featuremap',
            'expected_format_list': str(expect_format_list),
            'format': data_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_stride(strides, dim_s_h, dim_s_w, dim_s_n, dim_s_c):
    """
    check stride type and dim
    """
    if strides[dim_s_h] != strides[dim_s_w]:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'attr_name': 'strides value',
            'param1_name': 'strides[dim_s_h]',
            'param2_name': 'strides[dim_s_w]',
            'param1_value': str(strides[dim_s_h]),
            'param2_value': str(strides[dim_s_w])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if (strides[dim_s_n] != 1) or (strides[dim_s_c] != 1):
        error_manager_cube.raise_err_specific_user("depthwise_backprob_input",
                                                   "the N-dim and C-dim of stride must be equal to 1.")


# pylint: disable=locally-disabled, too-many-arguments
def _check_input_filter_shape(input_shape, output_shape, filter_shape, dim_n, dim_c, dim_w_c, dim_w_k):
    """
    check input and filter shape is valid
    """
    if input_shape[dim_n] != output_shape[dim_n]:
        dict_args = {
            'errCode': 'E64002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param1': 'input_shape[dim_n]',
            'param2': 'output_shape[dim_n]',
            'actual_value': str(input_shape[dim_n])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if input_shape[dim_c] != output_shape[dim_c]:
        dict_args = {
            'errCode': 'E64002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param1': 'input_shape[dim_c]',
            'param2': 'output_shape[dim_c]',
            'actual_value': str(input_shape[dim_c])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if (_ceil(input_shape[dim_c]) // BLOCK_SIZE) != (_ceil(filter_shape[dim_w_c]) // BLOCK_SIZE):

        dict_args = {
            'errCode': 'E64002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param1': '_ceil(input_shape[dim_c])',
            'param2': '(_ceil(filter_shape[dim_w_c]) // BLOCK_SIZE)',
            'actual_value': str(_ceil(input_shape[dim_c]))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if filter_shape[dim_w_k] != 1:
        dict_args = {
            'errCode': 'E64002',
            'op_name': 'depthwise_conv2d_backprop_input',
            'param1': 'filter_shape[dim_w_k]',
            'param2': '1',
            'actual_value': str(filter_shape[dim_w_k])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_dilations(dilations, dim_d_n, dim_d_c):
    """
    check dilations shape
    """
    if (dilations[dim_d_n] != 1) or (dilations[dim_d_c] != 1):
        dict_args = {
            'errCode': 'E62510',
            'op_name': 'depthwise_conv2d_backprop_input',
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_output_backprop(output_height, output_width, out_backprop_height, out_backprop_width):
    """
    check output and out_backprop shape is equal
    """
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
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
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
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_pad(pads):
    """
    check pad parameter
    """
    if len(pads) != 4:
        dict_args = {
            'errCode': 'E50001',
            'param': 'pads',
            'op_name': 'depthwise_conv2d_backprop_input',
            'expected_length': "4",
            'length': str(len(pads))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


# pylint: disable=too-many-statements, redefined-builtin
# pylint: disable=locally-disabled, too-many-locals, too-many-arguments, invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def depthwise_conv2d_backprop_input_d(filter,
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
    input_shape = input_grad.get("ori_shape")
    _check_input_size(input_size, input_shape)

    input_dtype = input_grad.get("dtype").lower()
    filter_shape = filter.get("ori_shape")
    filter_dtype = filter.get("dtype").lower()

    output_shape = out_backprop.get("ori_shape")
    output_dtype = out_backprop.get("dtype").lower()

    input_ori_format = input_grad.get('ori_format')
    _check_data_format(input_ori_format, ['NCHW', 'NHWC'])

    filter_ori_format = filter.get('ori_format')
    _check_data_format(filter_ori_format, ['HWCK', 'HWCN', 'NCHW'])

    dout_ori_format = out_backprop.get('ori_format')
    _check_data_format(dout_ori_format, ['NCHW', 'NHWC'])

    # index of the strides dimension
    dim_s_n, dim_s_c, dim_s_h, dim_s_w = 0, 1, 2, 3
    # index of the dilations dimension
    dim_d_n, dim_d_c, dim_d_h, dim_d_w = 0, 1, 2, 3
    # index of the out_backprop dimension
    dim_n, dim_c, _, _ = 0, 1, 2, 3
    # index of the filter dimension
    _, _, dim_w_c, dim_w_k = 0, 1, 2, 3

    if input_ori_format == 'NHWC':
        dim_s_n, dim_s_h, dim_s_w, dim_s_c = 0, 1, 2, 3
        dim_d_n, dim_d_h, dim_d_w, dim_d_c = 0, 1, 2, 3
        input_shape = [input_shape[0], input_shape[3], input_shape[1], input_shape[2]]
    if dout_ori_format == 'NHWC':
        output_shape = [output_shape[0], output_shape[3], output_shape[1], output_shape[2]]
    if filter_ori_format == "NCHW":
        filter_shape = [filter_shape[2], filter_shape[3], filter_shape[1], filter_shape[0]]
    _check_data_format(data_format, ['NCHW', 'NHWC'])

    # check if the parameter is valid
    _check_params(filter_shape, filter_dtype, "HWCK")
    _check_params(output_shape, output_dtype, "NCHW")
    _check_params(input_shape, input_dtype, "NCHW")
    para_check.check_shape(output_shape, min_rank=FEATURE_MAP_DIM, max_rank=FEATURE_MAP_DIM, param_name="out_backprop")
    para_check.check_shape(filter_shape, min_rank=FILTER_DIM, max_rank=FILTER_DIM, param_name="filter")
    para_check.check_shape(input_shape, min_rank=FEATURE_MAP_DIM, max_rank=FEATURE_MAP_DIM, param_name="input_grad")
    para_check.check_shape(strides, min_rank=STRIDES_DIM, max_rank=STRIDES_DIM, param_name="strides")
    para_check.check_shape(dilations, min_rank=DILATIONS_DIM, max_rank=DILATIONS_DIM, param_name="dilations")

    _check_stride(strides, dim_s_h, dim_s_w, dim_s_n, dim_s_c)
    _check_dilations(dilations, dim_d_n, dim_d_c)
    _check_input_filter_shape(input_shape, output_shape, filter_shape, dim_n, dim_c, dim_w_c, dim_w_k)
    # check pad parameter
    _check_pad(pads)

    # input parameters
    batch, input_channel, input_height, input_width = input_shape
    filter_height, filter_width, filter_channel, _ = filter_shape
    input_c1 = (input_channel + BLOCK_SIZE - 1) // BLOCK_SIZE
    stride_h, stride_w = strides[dim_s_h], strides[dim_s_w]
    dilation_h, dilation_w = dilations[dim_d_h], dilations[dim_d_w]
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

    _check_output_backprop(output_height, output_width, out_backprop_height, out_backprop_width)

    filter_shape = [_ceil(filter_channel) // BLOCK_SIZE, filter_height * filter_width, 1, BLOCK_SIZE, BLOCK_SIZE]
    filter_init = tvm.placeholder(filter_shape, dtype=filter_dtype, name='filter')

    output_shape = [batch, output_c1, 1, output_height, output_width, BLOCK_SIZE]
    dout = tvm.placeholder(output_shape, dtype=output_dtype, name='dout')

    input_shape = [batch, input_c1, 1, input_height, input_width, BLOCK_SIZE]
    res = depthwise_conv2d_backprop_input_d_compute(input_shape, filter_init, dout, [filter_height, filter_width],
                                                    strides, pads, kernel_name)

    sch = depthwise_conv2d_backprop_input_d_schedule(res)
    with tbe_platform.build_config:
        tvm.build_module.build(sch, [filter_init, dout, res], "cce", name=kernel_name)
