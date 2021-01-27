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
depthwise_conv2d_backprop_filter_d
"""

import te.platform as tbe_platform
from te.utils import para_check
from te.utils.error_manager import error_manager_util
from te.utils.error_manager import error_manager_cube
from te.lang.cce.te_compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_filter_d_compute
from tbe.dsl.static_schedule.depthwise_conv2d_schedule import depthwise_conv2d_backprop_filter_d_schedule
from te import tvm

BLOCK_SIZE = tbe_platform.BLOCK_REDUCE

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 4

# shape's dim of filter must be 4
FILTER_DIM = 4

# shape's dim of strides must be 4
STRIDES_DIM = 4

# shape's dim of dilation must be 4
DILATION_DIM = 4


def _ceil(x_size):
    """
    Return the least multiple of 16 integer number
    which is greater than or equal to x_size.
    """
    if BLOCK_SIZE:
        return ((x_size + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    dict_args = {'errCode': 'E67006', 'op_name': 'depthwise_conv2d_backprop_filter', 'param_name': 'BLOCK_SIZE'}
    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_shape(fmap_shape, dout_shape, filter_shape):
    """Check input shape."""
    fmap_n, fmap_c, _, _ = fmap_shape
    dout_n, dout_c, _, _ = dout_shape
    _, _, filter_c, filter_n = filter_shape

    if filter_n != 1:
        dict_args = {
            'errCode': 'E60000',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'param_name': 'filter_n',
            'expected_value': '1',
            'input_value': str(filter_n)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if fmap_c != dout_c:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'attr_name': 'channel value',
            'param1_name': 'fmap_c',
            'param2_name': 'dout_c',
            'param1_value': str(fmap_c),
            'param2_value': str(dout_c)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if fmap_n != dout_n:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'attr_name': 'channel value',
            'param1_name': 'fmap_n',
            'param2_name': 'dout_n',
            'param1_value': str(fmap_n),
            'param2_value': str(dout_n)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if fmap_c != filter_c:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'attr_name': 'channel value',
            'param1_name': 'fmap_c',
            'param2_name': 'filter_c',
            'param1_value': str(fmap_c),
            'param2_value': str(filter_c)
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


def _check_dilations(dilations, dim_n, dim_c, dim_h, dim_w):
    """
    check dilations dimension
    """
    if dilations[dim_n] != 1 or dilations[dim_c] != 1:
        error_manager_cube.raise_err_specific_user("depthwise_backprob_filter",
                                                   "dilation only support 1 in N axis and C axis.")
    if dilations[dim_h] != 1 or dilations[dim_w] != 1:
        dict_args = {
            'errCode': 'E60023',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'dilation_n': str(dilations[dim_n]),
            'dilation_c': str(dilations[dim_c])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


# pylint: disable=locally-disabled, too-many-locals, too-many-arguments,
def _check_stride(strides, dilations, dim_n, dim_c, dim_h, dim_w):
    """
    check stride type and dim
    """
    if strides[dim_n] != 1 or strides[dim_c] != 1:
        dict_args = {
            'errCode': 'E60023',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'dilation_n': str(dilations[dim_n]),
            'dilation_c': str(dilations[dim_c])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if strides[dim_h] != strides[dim_w]:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'attr_name': 'strides value',
            'param1_name': 'strides[dim_h]',
            'param2_name': 'strides[dim_w]',
            'param1_value': str(strides[dim_h]),
            'param2_value': str(strides[dim_w])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


# pylint: disable=invalid-name,
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def depthwise_conv2d_backprop_filter_d(input_fm,
                                       out_backprop,
                                       filter_grad,
                                       filter_size,
                                       strides,
                                       dilations=(1, 1, 1, 1),
                                       pads=(0, 0, 0, 0),
                                       data_format='NHWC',
                                       kernel_name="depthwise_conv2d_backprop_filter"):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution backward filter

    Parameters
    ----------
    input_fm : a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    out_backprop: a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    filter_grad : a dict.
        4-D origin shape of filter tensor [H, W, C, K],
        K is depthwise_multiplier, support float32.

    filter_size : a list/tuple of four ints
        1-D origin shape of filter tensor with [H, W, C, K],
        K is depthwise_multiplier, support int.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1].

    dilations : a list/tuple of four ints
        dilations size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1].

    pads : a list/tuple of four ints
        padding added to each dimension of the input.

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C].

    kernel_name : str
        cce kernel name

    Returns
    -------
    None
    """
    shape_in = input_fm.get('ori_shape')
    shape_w = filter_size
    shape_dout = out_backprop.get('ori_shape')
    in_dtype = input_fm.get('dtype')
    w_dtype = filter_grad.get('dtype')
    dout_dtype = out_backprop.get('dtype')

    _check_data_format(data_format, ['NCHW', 'NHWC'])
    if shape_w != filter_grad.get('ori_shape'):
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'attr_name': 'shape',
            'param1_name': 'shape_w',
            'param2_name': 'ori_shape',
            'param1_value': str(shape_w),
            'param2_value': str(filter_grad.get('ori_shape'))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    input_ori_format = input_fm.get('ori_format')
    _check_data_format(input_ori_format, ['NCHW', 'NHWC'])

    dout_ori_format = out_backprop.get('ori_format')
    _check_data_format(dout_ori_format, ['NCHW', 'NHWC'])

    filter_grad_ori_format = filter_grad.get('ori_format')
    _check_data_format(filter_grad_ori_format, ['HWCK', 'HWCN', 'NCHW'])

    if filter_grad_ori_format in ('NCHW', ):
        # NCHW to HWCK(HWCN)
        shape_w = (shape_w[2], shape_w[3], shape_w[1], shape_w[0])

    para_check.check_dtype(in_dtype.lower(), ('float16', ), param_name="input_fm")
    para_check.check_dtype(dout_dtype.lower(), ('float16', ), param_name="out_backprop")
    para_check.check_dtype(w_dtype.lower(), ('float32', ), param_name="filter_grad")

    para_check.check_shape(shape_in, min_rank=FEATURE_MAP_DIM, max_rank=FEATURE_MAP_DIM, param_name="input_fm")
    para_check.check_shape(shape_w, min_rank=FILTER_DIM, max_rank=FILTER_DIM, param_name="filter_grad")
    para_check.check_shape(shape_dout, min_rank=FEATURE_MAP_DIM, max_rank=FEATURE_MAP_DIM, param_name="out_backprop")
    para_check.check_shape(strides, min_rank=STRIDES_DIM, max_rank=STRIDES_DIM, param_name="strides")
    para_check.check_shape(dilations, min_rank=DILATION_DIM, max_rank=DILATION_DIM, param_name="dilations")

    # index of the origin dimension
    dim_n, dim_c, dim_h, dim_w = 0, 1, 2, 3  # NCHW
    if input_ori_format == 'NHWC':
        dim_n, dim_h, dim_w, dim_c = 0, 1, 2, 3
        shape_in = [shape_in[dim_n], shape_in[dim_c], shape_in[dim_h], shape_in[dim_w]]
    if dout_ori_format == 'NHWC':
        shape_dout = [shape_dout[0], shape_dout[3], shape_dout[1], shape_dout[2]]

    _check_shape(shape_in, shape_dout, shape_w)
    _check_dilations(dilations, dim_n, dim_c, dim_h, dim_w)
    _check_stride(strides, dilations, dim_n, dim_c, dim_h, dim_w)

    # check pad parameter
    if len(pads) != 4:
        dict_args = {
            'errCode': 'E50001',
            'param': 'pads',
            'op_name': 'depthwise_conv2d_backprop_filter',
            'expected_length': "4",
            'length': str(len(pads))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    n_size, c_size, h_size, w_size = shape_in
    shape_in = [n_size, _ceil(c_size) // BLOCK_SIZE, 1, h_size, w_size, BLOCK_SIZE]
    fmap_placeholder = tvm.placeholder(shape_in, dtype=in_dtype.lower(), name='fmap')
    n_size, c_size, h_size, w_size = shape_dout
    shape_dout = [n_size, _ceil(c_size) // BLOCK_SIZE, 1, h_size, w_size, BLOCK_SIZE]
    dout_placeholder = tvm.placeholder(shape_dout, dtype=dout_dtype.lower(), name='dout')

    h_size, w_size, _, _ = shape_w
    res = depthwise_conv2d_backprop_filter_d_compute(fmap_placeholder, dout_placeholder, h_size, w_size,
                                                     (strides[dim_h], strides[dim_w]), pads,
                                                     (dilations[dim_h], dilations[dim_w]), w_dtype.lower(), kernel_name)
    sch = depthwise_conv2d_backprop_filter_d_schedule(res)

    with tbe_platform.build_config:
        tvm.build_module.build(sch, [fmap_placeholder, dout_placeholder, res], "cce", name=kernel_name)
