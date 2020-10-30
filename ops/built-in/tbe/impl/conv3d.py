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
conv3d
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils.error_manager import error_manager_util
from te.lang.cce.te_compute import conv3d_compute
from te import tvm


BIAS_LENGTH = 1
# [strides_batch, strides_depth, strides_height,
#  strides_width, strides_channel]
STRIDE_LENGTH = 5

DILATION_LENGTH = 5
PADS_LENGTH = 6
# NDHWC or NCDHW
SHAPE_DIMS = 5


def _get_mad_dtype(w_dtype):
    """
    algorithm: Get the dtype of mad

    Parameters
    ----------
    w_dtype: The dtype of filter

    Returns
    -------
    mad dtype
    """
    mad_dtype = "float32"
    if w_dtype == 'int8':
        mad_dtype = "int32"
    elif tbe_platform.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES",
                                                      "Hi3796CV300CS"):
        mad_dtype = "float16"

    return mad_dtype


def _conv3d_compute(shape_fm,
                    shape_filter,
                    bias,
                    stride_dhw,
                    pads,
                    fmp_dtype,
                    w_dtype,
                    res_dtype,
                    kernel_name='conv3d'):
    """
    algorithm: compute conv3d

    Parameters
    ----------
    shape_fm: The shape of feature,
        A list/tuple of 'int' that has length `== 5`

    shape_filter: The shape of filter, a list of 'int' that has length `== 5`

    bias: A dict with keys(shape and dtype) or None
        An input bias tensor

    stride_dhw: A tuple/list of `ints` that has length `== 3`

    pads: A tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    fmp_dtype: The dtype of feature

    w_dtype: The dtype of filter

    res_dtype: The dtype of output

    kernel_name: Str
        Kernel name, default value is "conv3d"

    Returns
    -------
    list of tensor
    """
    batch, cin, fmp_d, fmp_h, fmp_w = shape_fm
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fmp_ndc1hwc0 = (batch, fmp_d, cin // fmp_block_k, fmp_h, fmp_w,
                          fmp_block_k)

    cout, cin, w_d, w_h, w_w = shape_filter
    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    w_block_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    shape_w_frac_z = (w_d * cin * w_h * w_w // w_block_k, cout // w_block_n,
                      w_block_n, w_block_k)

    mad_dtype = _get_mad_dtype(w_dtype)

    data = tvm.placeholder(shape_fmp_ndc1hwc0, name='Fmap', dtype=fmp_dtype)
    weight = tvm.placeholder(shape_w_frac_z, name='Filter', dtype=w_dtype)
    bias_tensor = None
    if bias:
        bias_tensor = tvm.placeholder((cout, ),
                                      name='bias_tensor',
                                      dtype=res_dtype)
    conv3d_dict = {
        "bias_tensor": bias_tensor,
        "pads": pads,
        "shape_filter_ncdhw": shape_filter,
        "stride_dhw": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name
    }
    conv_res = conv3d_compute.conv3d(data, weight, conv3d_dict)
    if bias:
        tensor_list = [data, weight, bias_tensor, conv_res]
    else:
        tensor_list = [data, weight, conv_res]

    return tensor_list


def _check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype):
    """
    algorithm: Check the input params of conv3d

    Parameters
    ----------

    fmp_dtype: The dtype of feature

    w_dtype: The dtype of filter

    res_dtype: The dtype of output

    Returns
    -------
    None
    """
    para_check.check_dtype_rule(fmp_dtype, ('float16', ))
    para_check.check_dtype_rule(w_dtype, ('float16', ))
    para_check.check_dtype_rule(res_dtype, ('float16', ))


def _format_normalize(fmp_format, w_format, fmp_shape, w_shape, strides,
                     dilations):
    """
    algorithm: unified format

    Parameters
    ----------
    fmp_format: The data format of the input feature

    w_format: The data format of the input filter

    fmp_shape: The shape of feature
        A list/tuple of 'int' that has length `== 5`

    w_shape: The shape of filter, a list of 'int' that has length `== 5`

    strides: A tuple/list of `ints` that has length `== 5`

    dilations: A tuple/list of 5 integers
        Dilation on D/H/W, format sensitive
        Dilations in the batch and depth dimensions must be 1

    Returns
    -------
    shape_fm, shape_filter, stride_dhw, dilation_hw
    """
    if fmp_format == "NCDHW":
        shape_fm = list(fmp_shape)
        stride_dhw = strides[2:]
        dilation_hw = dilations[3:]
    elif fmp_format == "NDHWC":
        shape_fm = [
            fmp_shape[0], fmp_shape[4], fmp_shape[1], fmp_shape[2],
            fmp_shape[3]
        ]
        stride_dhw = strides[1:4]
        dilation_hw = dilations[2:4]
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'input',
            'expected_format_list': '[{}, {}]'.format('NCDHW', 'NDHWC'),
            'format': fmp_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if w_format == "NCDHW":
        shape_filter = list(w_shape)
    elif w_format == "NDHWC":
        shape_filter = [
            w_shape[0], w_shape[4], w_shape[1], w_shape[2], w_shape[3]
        ]
    elif w_format == "DHWCN":
        shape_filter = [
            w_shape[4], w_shape[3], w_shape[0], w_shape[1], w_shape[2]
        ]
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': '[{}, {}, {}]'
                                    .format('NCDHW', 'NDHWC', 'DHWCN'),
            'format': w_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return shape_fm, shape_filter, stride_dhw, dilation_hw


def _check_input_param(fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype,
                       fmp_format, w_format, bias, strides, pads, dilations):
    """
    algorithm: Check the input params of conv3d

    Parameters
    ----------
    fmp_shape: The shape of feature
        A list/tuple of 'int' that has length `== 5`

    w_shape: The shape of filter
        A list/tuple of 'int' that has length `== 5`

    fmp_dtype: The dtype of feature

    w_dtype: The dtype of filter

    res_dtype: The dtype of output

    fmp_format: The data format of the input feature

    w_format: The data format of the input filter

    bias: A dict with keys(shape and dtype) or None
        input bias tensor

    strides: A list/tuple of `ints` that has length `== 5`

    pads: A list/tuple of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A list/tuple of 5 integers
        Dilation on D/H/W, format sensitive
        Dilations in the batch and depth dimensions must be 1

    Returns
    -------
    """
    if bias:
        bias_dtype = bias.get("dtype")
        para_check.check_dtype_rule(bias_dtype, ('float16', ))
        bias_shape = bias.get("ori_shape")
        if len(bias_shape) != BIAS_LENGTH:
            dict_args = {
                'errCode': 'E60006',
                'param_name': 'bias',
                'expected_length': '1',
                'length': '{}'.format(len(bias_shape))
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))
    if len(strides) != STRIDE_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'strides',
            'expected_length': '5',
            'length': '{}'.format(len(strides))
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    if len(dilations) != DILATION_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'dilations',
            'expected_length': '5',
            'length': '{}'.format(len(dilations))
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    # check dilations for it1
    if len(set(dilations)) != 1 or dilations[2] != 1:
        dict_args = {
            'errCode': 'E62001',
            'dilation_h': str(dilations[2]),
            'dilation_w': str(dilations[3]),
            'dilation_d': str(dilations[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if len(pads) != PADS_LENGTH:
        dict_args = {
            'errCode': 'E62501',
            'param_name': 'pads',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    para_check.check_shape_rule(fmp_shape, min_dim=SHAPE_DIMS,
                                max_dim=SHAPE_DIMS)
    para_check.check_shape_rule(w_shape, min_dim=SHAPE_DIMS,
                                max_dim=SHAPE_DIMS)

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, dilation_hw = _format_normalize(
        fmp_format, w_format, fmp_shape, w_shape, strides, dilations)

    _check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype)

    conv3d_compute.check_conv3d_shape(shape_fm, shape_filter, pads,
                                      stride_dhw, fmp_dtype, w_dtype)

    return shape_fm, shape_filter, stride_dhw, dilation_hw


@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_STR,
    para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def conv3d(fmap,
           weight,
           bias,
           offset_w,
           output,
           strides,
           pads,
           dilations=(1, 1, 1, 1, 1),
           groups=1,
           data_format="NDHWC",
           offset_x=0,
           kernel_name="conv3d"):
    """
    algorithm: conv3d

    Parameters
    ----------
    fmap: A dict with keys(shape and dtype)
        Input 5d feature map tensor

    weight: A dict with keys(shape and dtype)
        Input 5d weight tensor

    bias: A dict with keys(shape and dtype) or None
        Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
        Input offset_w tensor

    output: A dict with keys(shape and dtype)
        Output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers, format sensitive
        [strides_batch, strides_depth, strides_height, strides_width, strides_channel]

    pads: A tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
        Dilation on D/H/W, format sensitive, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
        Default value is 1

    data_format: The data format of the input and output data
        Default format is "NDHWC"

    offset_x: Int
        Input offset_x value, default value is 0

    kernel_name: Str
        Kernel name, default value is "conv3d"

    Returns
    -------
    None
    """
    def _conv3d_achieve_with_tvm():
        tensor_list = _conv3d_compute(shape_fm,
                                      shape_filter,
                                      bias,
                                      stride_dhw,
                                      pads,
                                      fmp_dtype,
                                      w_dtype,
                                      res_dtype,
                                      kernel_name=kernel_name)

        with tvm.target.cce():
            sch = tbe.auto_schedule(tensor_list[-1])

        config = {"name": kernel_name, "tensor_list": tensor_list}
        tbe.cce_build_code(sch, config)

    fmp_shape = fmap.get("ori_shape")
    fmp_dtype = fmap.get("dtype")
    fmp_format = data_format
    w_shape = weight.get("ori_shape")
    w_dtype = weight.get("dtype")
    w_format = weight.get("ori_format")
    res_dtype = output.get("dtype")

    fmp_dtype = fmp_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, _ = _check_input_param(
        fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype, fmp_format,
        w_format, bias, strides, pads, dilations)

    pads = list(pads)
    stride_dhw = list(stride_dhw)

    # C and Cout align 16
    shape_fm = list(shape_fm)
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fm[1] = (
        (shape_fm[1] + fmp_block_k - 1) // fmp_block_k) * fmp_block_k
    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    shape_filter = list(shape_filter)
    shape_filter[1] = (
        (shape_filter[1] + w_block_k - 1) // w_block_k) * w_block_k
    w_block_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    shape_filter[0] = (
        (shape_filter[0] + w_block_n - 1) // w_block_n) * w_block_n

    _conv3d_achieve_with_tvm()
