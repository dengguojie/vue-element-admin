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


BIAS_LENGTH = 1
# [strides_batch, strides_depth, strides_height,
#  strides_width, strides_channel]
STRIDE_LENGTH = 5

DILATION_LENGTH = 5
PADS_LENGTH = 6
# NDHWC or NCDHW
SHAPE_DIMS = 5
C0 = 16
L1FUSION_INPUT_CTR = 2

FMAP_TARGET_FORMAT = "NCDHW"
FMAP_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
FILTER_TARGET_FORMAT = "NCDHW"
FILTER_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC", "DHWCN"]

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
        overlap_d = -1 if filter_shape[2] <= strides_formated[2] else 0
        overlap_h = -1 if filter_shape[3] <= strides_formated[3] else 0
        overlap_w = -1 if filter_shape[4] <= strides_formated[4] else 0

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
                                                FILTER_TARGET_FORMAT,
                                                weight.get("ori_shape"),
                                                FILTER_FORMAT_WHITE_LIST)
    if filter_shape is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': ",".join(FILTER_FORMAT_WHITE_LIST),
            'format': weight.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    strides_formated = _transform_shape_with_format(fmap.get("ori_format"),
                                                    FMAP_TARGET_FORMAT,
                                                    strides,
                                                    FMAP_FORMAT_WHITE_LIST)
    if strides_formated is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'strides',
            'expected_format_list': ",".join(FMAP_FORMAT_WHITE_LIST),
            'format': fmap.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              L1FUSION_INPUT_CTR,
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
                                                      "Hi3796CV300CS"):
        mad_dtype = "float16"

    return mad_dtype


def _lcm(param1, param2):
    """
    calculate least common multiple
    """
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, param1 % param2

    return temp // param2


def _calculate_group(fmap_c, cout, groups):
    """
    calculate groups Parameter
    """
    mag_factor0 = _lcm(fmap_c // groups, C0) // (fmap_c // groups)
    mag_factor1 = _lcm(cout // groups, C0) // (cout // groups)
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin1_g = (mag_factor * fmap_c // groups + C0 - 1) // C0
    cout_g = (mag_factor * cout // groups + C0 - 1) // C0 * C0

    group_dict = {"real_g": (groups + mag_factor - 1) // mag_factor,
                  "mag_factor": mag_factor,
                  "cin1_g": cin1_g,
                  "cout_g": cout_g,
                  "cin_ori": fmap_c,
                  "cout_ori": cout}

    return group_dict


def _conv3d_compute(shape_fm,
                    shape_filter,
                    bias,
                    stride_dhw,
                    pads,
                    fmp_dtype,
                    w_dtype,
                    res_dtype,
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

    kernel_name: Str
        Kernel name, default value is "conv3d"

    group_dict: Dict
        Group convolution related information

    Returns
    -------
    list of tensor
    """
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    cout_ori = group_dict["cout_ori"]
    
    batch, cin, fmp_d, fmp_h, fmp_w = shape_fm
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fmp_ndc1hwc0 = (batch, fmp_d, (cin + fmp_block_k - 1) // fmp_block_k, fmp_h, fmp_w,
                          fmp_block_k)

    cout, cin, w_d, w_h, w_w = shape_filter
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
    conv3d_dict = {
        "bias_tensor": bias_tensor,
        "pads": pads,
        "shape_filter_ncdhw": shape_filter,
        "stride_dhw": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }
    conv_res = conv3d_compute.conv3d(data, weight, conv3d_dict, fusion_flag=False)

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
                                            FMAP_TARGET_FORMAT,
                                            fmp_shape,
                                            FMAP_FORMAT_WHITE_LIST)
    stride_full = _transform_shape_with_format(fmp_format,
                                               FMAP_TARGET_FORMAT,
                                               strides,
                                               FMAP_FORMAT_WHITE_LIST)
    dilation_full = _transform_shape_with_format(fmp_format,
                                                 FMAP_TARGET_FORMAT,
                                                 dilations,
                                                 FMAP_FORMAT_WHITE_LIST)

    if shape_fm is None or stride_full is None or dilation_full is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'input',
            'expected_format_list': ",".join(FMAP_FORMAT_WHITE_LIST),
            'format': fmp_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    shape_filter = _transform_shape_with_format(w_format,
                                                FILTER_TARGET_FORMAT,
                                                w_shape,
                                                FILTER_FORMAT_WHITE_LIST)
    if shape_filter is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': ",".join(FILTER_FORMAT_WHITE_LIST),
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
    shape_fm, shape_filter, stride_dhw, dilation_dhw = _format_normalize(
        fmp_format, w_format, fmp_shape, w_shape, strides, dilations)
    # check group
    if groups <= 0 or groups > shape_fm[1] or groups > shape_filter[0]:
        dict_args = {
            'errCode': 'E50060',
            'description': "Group must not be larger than x channel and filter channel"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if shape_fm[1] % groups != 0 or shape_filter[0] % groups != 0:
        dict_args = {
            'errCode': 'E50060',
            'description': "Feature map's or filter's channel must be divisible by group"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    group_dict = _calculate_group(shape_fm[1], shape_filter[0], groups)

    _check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype)
    # 16 aligned
    conv3d_compute.check_conv3d_shape(shape_fm, shape_filter, pads,
                                      stride_dhw, dilation_dhw, fmp_dtype, w_dtype, groups)

    return shape_fm, shape_filter, stride_dhw, dilation_dhw, group_dict


def cal_input_param(fmap, weight, bias_tensor, strides, pads, dilations, groups, data_format, kernel_name):
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
    shape_fmap, shape_filter, stride_dhw, dilation_hw = _format_normalize(
        data_format, w_format, shape_fmap, shape_filter, strides, dilations)

    group_dict = _calculate_group(shape_fmap[1], shape_filter[0], groups)

    conv3d_dict = {
        "bias_tensor": bias_tensor,
        "pads": pads,
        "shape_filter_ncdhw": shape_filter,
        "stride_dhw": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }
    return conv3d_dict


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
    shape_fm, shape_filter, stride_dhw, _, group_dict = _check_input_param(
        fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype, fmp_format,
        w_format, bias, strides, pads, dilations, groups)

    pads = list(pads)
    stride_dhw = list(stride_dhw)

    # C and Cout align 16
    shape_fm = list(shape_fm)
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fm[1] = ((shape_fm[1] + fmp_block_k - 1) // fmp_block_k) * fmp_block_k
    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    shape_filter = list(shape_filter)
    shape_filter[1] = ((shape_filter[1] + w_block_k - 1) // w_block_k) * w_block_k
    w_block_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    shape_filter[0] = ((shape_filter[0] + w_block_n - 1) // w_block_n) * w_block_n

    tensor_list = _conv3d_compute(shape_fm,
                                  shape_filter,
                                  bias,
                                  stride_dhw,
                                  pads,
                                  fmp_dtype,
                                  w_dtype,
                                  res_dtype,
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
    conv3d_dict = cal_input_param(data, weight, bias, strides, pads, dilations, groups, data_format, kernel_name)

    res = conv3d_compute.conv3d(data, weight, conv3d_dict)

    return res
