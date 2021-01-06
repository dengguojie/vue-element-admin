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
from te import tvm
from te.lang.cce.te_compute import conv3d_compute
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check
from te.utils.error_manager import error_manager_util
from impl.util import util_select_op_base
from impl.util import util_common

_BIAS_LENGTH = 1
# [strides_batch, strides_depth, strides_height,
#  strides_width, strides_channel]
_STRIDE_LENGTH = 5

_DILATION_LENGTH = 5
_PADS_LENGTH = 6
# NDHWC or NCDHW
_SHAPE_DIMS = 5
_C0 = 16
_L1FUSION_INPUT_CTR = 2

_FMAP_TARGET_FORMAT = "NCDHW"
_FMAP_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
_FILTER_TARGET_FORMAT = "NCDHW"
_FILTER_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC", "DHWCN"]

def _transform_shape_with_format(src_format, to_format, ori_shape, format_white_list):
    # input format is not expected
    if ((src_format not in format_white_list) or
        (to_format not in format_white_list)):
        return None
    # need not to transform
    if src_format == to_format:
        return ori_shape
    res_shape = [1 for _ in range(len(to_format))]
    for i in range(len(to_format)):
        for j in range(len(src_format)):
            if to_format[i] == src_format[j]:
                res_shape[i] = ori_shape[j]
    return res_shape

def get_op_support_info(fmap,
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
                        kernel_name="conv3d",
                        op_slice_info=""):
    """
    algorithm: get_op_support_info

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

    op_slice_info: Str
        Default value is ""

    Returns
    -------
    op_cal_info_in_json: A dict with keys(split_maps, reduce_maps, l1_fusion_enable
                         and min_tbe_l1_space)
    """
    def _get_slice_info():
        overlap_d = -1 if (filter_d - 1) * dilation_d + 1 <= strides_d else 0
        overlap_h = -1 if (filter_h - 1) * dilation_h + 1 <= strides_h else 0
        overlap_w = -1 if (filter_w - 1) * dilation_w + 1 <= strides_w else 0

        axis_split_matrix = []
        axis_reduce_list = []
        if fm_format == "NDC1HWC0":
            # cut N
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
                                    util_select_op_base.SplitOutput([0, [0]])])
            # cut D
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [1], [overlap_d], [overlap_d]]),
                                    util_select_op_base.SplitOutput([0, [1]])])
            # cut H
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [3], [overlap_h], [overlap_h]]),
                                    util_select_op_base.SplitOutput([0, [3]])])
            # cut W
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [4], [overlap_w], [overlap_w]]),
                                    util_select_op_base.SplitOutput([0, [4]])])
            # cut Cout
            if bias:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [1], [-1], [-1]], [2, [0], [-1], [-1]]),
                    util_select_op_base.SplitOutput([0, [2]])]
                )
            else:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [1], [-1], [-1]]),
                    util_select_op_base.SplitOutput([0, [2]])]
                )
            axis_reduce_list = None
        else:
            axis_split_matrix = None
            axis_reduce_list = None

        return axis_split_matrix, axis_reduce_list

    fm_format = fmap.get("format")
    filter_shape = _transform_shape_with_format(weight.get("ori_format"),
                                                _FILTER_TARGET_FORMAT,
                                                weight.get("ori_shape"),
                                                _FILTER_FORMAT_WHITE_LIST)
    if filter_shape is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': ",".join(_FILTER_FORMAT_WHITE_LIST),
            'format': weight.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    strides_formated = _transform_shape_with_format(fmap.get("ori_format"),
                                                    _FMAP_TARGET_FORMAT,
                                                    strides,
                                                    _FMAP_FORMAT_WHITE_LIST)
    if strides_formated is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'strides',
            'expected_format_list': ",".join(_FMAP_FORMAT_WHITE_LIST),
            'format': fmap.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    dilations_formated = _transform_shape_with_format(fmap.get("ori_format"),
                                                      _FMAP_TARGET_FORMAT,
                                                      dilations,
                                                      _FMAP_FORMAT_WHITE_LIST)

    _, _, filter_d, filter_h, filter_w = filter_shape
    _, strides_d, strides_h, strides_w, _ = strides_formated
    _, dilation_d, dilation_h, dilation_w, _ = dilations_formated

    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              _L1FUSION_INPUT_CTR,
                                                              0)
    return op_cal_info_in_json

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
                                                      "Hi3796CV300CS",
                                                      "SD3403"):
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
                    dilation_dhw=None,
                    group_dict=None,
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

    dilation_dhw: A tuple/list of `ints` that has length `==3`

    group_dict: Dict
        Group convolution related information

    kernel_name: Str
        Kernel name, default value is "conv3d"

    Returns
    -------
    list of tensor
    """
    if dilation_dhw is None:
        dilation_dhw = [1, 1, 1]
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    cout_ori = group_dict["cout_ori"]
    
    # C and Cout align 16
    shape_fm = list(shape_fm)
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fm[1] = ((shape_fm[1] + fmp_block_k - 1) // fmp_block_k) * fmp_block_k
    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    shape_filter = list(shape_filter)
    shape_filter[1] = ((shape_filter[1] + w_block_k - 1) // w_block_k) * w_block_k
    w_block_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    shape_filter[0] = ((shape_filter[0] + w_block_n - 1) // w_block_n) * w_block_n
    
    batch, cin, fmp_d, fmp_h, fmp_w = shape_fm
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fmp_ndc1hwc0 = (batch, fmp_d, cin // fmp_block_k, fmp_h, fmp_w, fmp_block_k)

    _, _, w_d, w_h, w_w = shape_filter
    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    w_block_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]


    shape_w_frac_z = (real_g * w_d * cin1_g * w_h * w_w, cout_g // w_block_n,
                      w_block_n, w_block_k)

    mad_dtype = _get_mad_dtype(w_dtype)

    data = tvm.placeholder(shape_fmp_ndc1hwc0, name='Fmap', dtype=fmp_dtype)
    weight = tvm.placeholder(shape_w_frac_z, name='Filter', dtype=w_dtype)
    
    bias_tensor = None
    if bias is not None:
        bias_tensor = tvm.placeholder((cout_ori,),
                                      name='bias_tensor',
                                      dtype=res_dtype)
    para_dict = {
        "dsl_flag": False,
        "bias_tensor": bias_tensor,
        "pads": pads,
        "strides": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "dilations": dilation_dhw
    }

    conv_res = conv3d_compute.conv3d(data, weight, shape_filter, para_dict)

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
    para_check.check_dtype_rule(fmp_dtype, ('float16'), "fmap")
    para_check.check_dtype_rule(w_dtype, ('float16'), "filter")
    para_check.check_dtype_rule(res_dtype, ('float16'), "output")


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
    shape_fm, shape_filter, stride_dhw, dilation_dhw
    """
    shape_fm = _transform_shape_with_format(fmp_format,
                                            _FMAP_TARGET_FORMAT,
                                            fmp_shape,
                                            _FMAP_FORMAT_WHITE_LIST)
    stride_full = _transform_shape_with_format(fmp_format,
                                               _FMAP_TARGET_FORMAT,
                                               strides,
                                               _FMAP_FORMAT_WHITE_LIST)
    dilation_full = _transform_shape_with_format(fmp_format,
                                                 _FMAP_TARGET_FORMAT,
                                                 dilations,
                                                 _FMAP_FORMAT_WHITE_LIST)

    if shape_fm is None or stride_full is None or dilation_full is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'input',
            'expected_format_list': ",".join(_FMAP_FORMAT_WHITE_LIST),
            'format': fmp_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    shape_filter = _transform_shape_with_format(w_format,
                                                _FILTER_TARGET_FORMAT,
                                                w_shape,
                                                _FILTER_FORMAT_WHITE_LIST)
    if shape_filter is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': ",".join(_FILTER_FORMAT_WHITE_LIST),
            'format': w_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    return shape_fm, shape_filter, stride_full[2:], dilation_full[2:]


def _check_input_param(fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype,
                       fmp_format, w_format, bias, strides, pads, dilations,
                       groups):
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

    groups: int
        Group convolution parameter

    Returns
    -------
    """
    if bias:
        bias_dtype = bias.get("dtype")
        para_check.check_dtype_rule(bias_dtype, ('float16'), "bias")
        bias_shape = bias.get("ori_shape")
        if len(bias_shape) != _BIAS_LENGTH:
            dict_args = {
                'errCode': 'E60006',
                'param_name': 'bias',
                'expected_length': '1',
                'length': '{}'.format(len(bias_shape))
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))
    if len(strides) != _STRIDE_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'strides',
            'expected_length': '5',
            'length': '{}'.format(len(strides))
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    if len(dilations) != _DILATION_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'dilations',
            'expected_length': '5',
            'length': '{}'.format(len(dilations))
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if len(pads) != _PADS_LENGTH:
        dict_args = {
            'errCode': 'E62501',
            'param_name': 'pads',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    para_check.check_shape_rule(fmp_shape, min_dim=_SHAPE_DIMS,
                                max_dim=_SHAPE_DIMS)
    para_check.check_shape_rule(w_shape, min_dim=_SHAPE_DIMS,
                                max_dim=_SHAPE_DIMS)

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, dilation_dhw = _format_normalize(
        fmp_format, w_format, fmp_shape, w_shape, strides, dilations)
    dilation_d, _, _ = dilation_dhw
    if dilation_d != 1:
        dict_args = {
            'errCode': 'E60038',
            'desc': 'dilation in D dimension only supports 1',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    group_dict = util_common.calculate_group(shape_fm[1], shape_filter[0], groups, _C0, _C0)

    _check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype)
    
    _check_groups_validation(shape_fm[1], shape_filter[1], groups)

    return shape_fm, shape_filter, stride_dhw, dilation_dhw, group_dict

def _check_groups_validation(fmap_cin, filter_cin, groups):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------
    fmap_cin: the C channel input of the feature map

    filter_cin: the C channel input of the filter

    groups: The groups for group convolution

    Returns
    -------
    None
    """
    if fmap_cin != filter_cin * groups:
        dict_args = {
            'errCode': 'E60010',
            'channel_of_x': str(fmap_cin),
            'channel_of_filter': str(filter_cin * groups)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

def _cal_input_param(fmap, weight, bias_tensor, strides, pads, dilations, groups, data_format, kernel_name):
    """
    to calculate fusion param
    """
    shape_fmap = []
    for i in fmap.op.attrs['ori_shape']:
        shape_fmap.append(i.value)

    shape_filter = []
    for i in weight.op.attrs['ori_shape']:
        shape_filter.append(i.value)

    pos_d = data_format.find('D')
    pos_h = data_format.find('H')
    pos_w = data_format.find('W')

    res_dtype = 'float16'
    mad_dtype = _get_mad_dtype(weight.dtype)

    w_format = weight.op.attrs['ori_format'].value
    # NCDHW
    shape_fmap, shape_filter, stride_dhw, dilation_dhw = _format_normalize(
        data_format, w_format, shape_fmap, shape_filter, strides, dilations)

    _check_groups_validation(shape_fmap[1], shape_filter[1], groups)

    group_dict = util_common.calculate_group(shape_fmap[1], shape_filter[0], groups, _C0, _C0)

    para_dict = {
        "dsl_flag": True,
        "bias_tensor": bias_tensor,
        "pads": pads,
        "strides": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "dilations": dilation_dhw
    }

    return para_dict, shape_filter


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
    shape_fm, shape_filter, stride_dhw, dilation_dhw, group_dict = _check_input_param(
        fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype, fmp_format,
        w_format, bias, strides, pads, dilations, groups)

    pads = list(pads)
    stride_dhw = list(stride_dhw)
    dilations_dhw = list(dilation_dhw)

    tensor_list = _conv3d_compute(shape_fm,
                                  shape_filter,
                                  bias,
                                  stride_dhw,
                                  pads,
                                  fmp_dtype,
                                  w_dtype,
                                  res_dtype,
                                  dilation_dhw=dilation_dhw,
                                  group_dict=group_dict,
                                  kernel_name=kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(tensor_list[-1])

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)


@fusion_manager.register("conv3d")
def conv3d_fusion_compute(data,
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
    """
    para_dict, filter_size = _cal_input_param(data, weight, bias, strides, pads, dilations, groups, data_format, kernel_name)

    res = conv3d_compute.conv3d(data, weight, filter_size, para_dict)

    return res
