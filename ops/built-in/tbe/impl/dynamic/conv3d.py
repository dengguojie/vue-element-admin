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
dynamic conv3d
"""
from __future__ import absolute_import

import math
import warnings

from impl.util import util_common
from impl.util import util_conv3d
from impl.util import util_cube_dynamic
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_register


# [strides_batch, strides_depth, strides_height,
#  strides_width, strides_channel]
STRIDE_LENGTH = 5
DILATION_LENGTH = 5
PADS_LENGTH = 6
# NDHWC or NCDHW
SHAPE_DIMS = 5
FORMAT_5D_DIMS = 5
# NDC1HWC0
FORMAT_6D_DIMS = 6
N_DIM_6D = 0
D_DIM_6D = 1
H_DIM_6D = 3
W_DIM_6D = 4
C0 = 16
HW_MIN = 1
HW_MAX = 4096
PAD_MIN = 0
PAD_MAX = 255
# filterD must be in [1,255]
FILTER_DHW_MIN = 1
FILTER_DHW_MAX = 255
# pad must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255
# stride must be in [1,63]
STRIDE_MIN = 1
STRIDE_MAX = 63

# fmap H and W must be in [1, 4096]
FMAP_HW_MIN = 1
FMAP_HW_MAX = 4096
MAX_SHAPE_NUM = 2 ** 31 - 1

FMAP_TARGET_FORMAT = "NCDHW"
FMAP_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
FILTER_TARGET_FORMAT = "NCDHW"
FILTER_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC", "DHWCN"]
VALID_DTYPE = ("float16",)
DYNAMIC_FLAG = -1
RANGE_DIM_LEN = 2
L1FUSION_INPUT_CTR = 2
DYNAMIC_RANK_FLAG = [-2]
_OP_TYPE = "conv3d"

_DTYPE_SIZE = {"int32": 4, "float32": 4, "float16": 2,
               "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
_ALIGN_BYTE = 32
_DEFAULT_FP16_SIZE = 2
_DEFAULT_L1_HO_LEN = 2
_BINARY_SEARCH_NUM = 2

# generalize error json
LOWER_LIST = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": ["lower_limit"]}}]
UPPER_LIST = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": ["upper_limit"]}}]
UNSUPPORT_LIST = [{"result": "UNSUPPORTED"}]


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
    op_cal_info_in_json: A dict with keys(split_maps, reduce_maps, l1_fusion_enable and min_tbe_l1_space)
    """
    def _get_slice_info():
        overlap_d = -1 if (filter_d == 1 and strides_d == 1) else 0
        overlap_h = -1 if (filter_h == 1 and strides_h == 1) else 0
        overlap_w = -1 if (filter_w == 1 and strides_w == 1) else 0

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
    filter_shape = util_conv3d.transform_shape_with_format(weight.get("ori_format"),
                                                           FILTER_TARGET_FORMAT,
                                                           weight.get("ori_shape"),
                                                           FILTER_FORMAT_WHITE_LIST)
    if filter_shape is None:
        error_manager_cube.raise_err_one_para("E62306", "Conv3d", "filter format should be NDHWC/NCDHW/DHWCN")

    strides_formated = util_conv3d.transform_shape_with_format(fmap.get("ori_format"),
                                                               FMAP_TARGET_FORMAT,
                                                               strides,
                                                               FMAP_FORMAT_WHITE_LIST)
    if strides_formated is None:
        error_manager_cube.raise_err_one_para("E62306", "Conv3d", "data format should be NDHWC or NCDHW")

    _, _, filter_d, filter_h, filter_w = filter_shape
    _, strides_d, strides_h, strides_w, _ = strides_formated

    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              L1FUSION_INPUT_CTR,
                                                              None)
    return op_cal_info_in_json


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
        error_manager_cube.raise_err_scene_equal_limitation("conv3d", 'channel of x',
                                                  'the product of the filter_channel and groups')


def _common_check(shape_filter, stride_dhw):
    _, _, filter_d, filter_h, filter_w = shape_filter
    stride_d, stride_h, stride_w = stride_dhw
    if filter_d < FILTER_DHW_MIN or filter_d > FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'weight', 'D',
            '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX), str(filter_d))

    if stride_d < STRIDE_MIN or stride_d > STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'D',
            '[{}, {}]'.format(STRIDE_MIN, STRIDE_MAX), str(stride_d))

    if filter_h < FILTER_DHW_MIN or filter_h > FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'filter', 'H',
            '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX), str(filter_h))

    if stride_h < STRIDE_MIN or stride_h > STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'H',
            '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX), str(stride_h))

    if filter_w < FILTER_DHW_MIN or filter_w > FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'filter', 'W',
            '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX), str(filter_w))

    if stride_w < STRIDE_MIN or stride_w > STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'W',
            '[{}, {}]'.format(STRIDE_MIN, STRIDE_MAX), str(stride_w))


def _check_d_dimension(fmap_d, filter_d, pad_d, dilation_d):
    filter_dilated_d = (filter_d - 1) * dilation_d + 1
    if fmap_d != DYNAMIC_FLAG and ((fmap_d + pad_d[0] + pad_d[1]) < filter_dilated_d):
        error_manager_cube.raise_err_three_paras("E62507", "conv3d", "D",
            str(filter_dilated_d), str(fmap_d + pad_d[0] + pad_d[1]))

    if pad_d[0] < PAD_MIN or pad_d[1] < PAD_MIN or pad_d[0] > PAD_MAX or pad_d[1] > PAD_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'D',
            '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'pad_d[0] = {}, pad_d[1] = {}'.format(pad_d[0], pad_d[1]))

    if pad_d[0] >= filter_dilated_d or pad_d[1] >= filter_dilated_d:
        error_manager_cube.raise_err_specific_user("conv3d",
            "the depth of pad can not be less than shape_filter's, \
             actual are {} and {}".format(pad_d[0], pad_d[1]))


def _check_h_dimension(fmap_h, filter_h, pad_h, dilation_h):
    filter_dilated_h = (filter_h - 1) * dilation_h + 1
    if pad_h[0] < PAD_MIN or pad_h[1] < PAD_MIN or pad_h[0] > PAD_MAX or pad_h[1] > PAD_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'H',
            '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'pad_h[0] = {}, pad_h[1] = {}'.format(pad_h[0], pad_h[1]))

    if fmap_h != DYNAMIC_FLAG and ((fmap_h + pad_h[0] + pad_h[1]) < filter_dilated_h):
        error_manager_cube.raise_err_three_paras("E62507", "conv3d", "H",
            str(filter_dilated_h), str(fmap_h + pad_h[0] + pad_h[1]))

    if pad_h[0] >= filter_dilated_h or pad_h[1] >= filter_dilated_h:
        error_manager_cube.raise_err_specific_user("conv3d",
            "the height of pad can not be less than shape_filter's, \
             actual are {} and {}".format(pad_h[0], pad_h[1]))


def _check_w_dimension(fmap_w, filter_w, pad_w, dilation_w):
    filter_dilated_w = (filter_w - 1) * dilation_w + 1
    if pad_w[0] < PAD_MIN or pad_w[1] < PAD_MIN or pad_w[0] > PAD_MAX or pad_w[1] > PAD_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'W',
            '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'pad_w[0] = {}, pad_w[1] = {}'.format(pad_w[0], pad_w[1]))

    if fmap_w != DYNAMIC_FLAG and (filter_dilated_w > (fmap_w + pad_w[0] + pad_w[1])):
        error_manager_cube.raise_err_three_paras("E62507", "conv3d", "W",
            str(filter_dilated_w), str(fmap_w + pad_w[0] + pad_w[1]))

    if pad_w[0] >= filter_dilated_w or pad_w[1] >= filter_dilated_w:
        error_manager_cube.raise_err_specific_user("conv3d",
            "the width of pad can not be less than shape_filter's, \
            actual are {} and {}".format(pad_w[0], pad_w[1]))


def _check_conv3d_shape(shape_fm, shape_filter, pads, stride_dhw, dilation_dhw,
                       fmp_dtype, w_dtype, out_range=None):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------
    shape_fm: the shape of feature, format is 'NCDHW'.
        a list/tuple of 'int' that has length `== 5`

    shape_filter: the shape of filter, format is 'NCDHW'.
        a list of 'int' that has length `== 5`

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    stride_dhw: A list of `ints` that has length `== 3`

    dilation_dhw: A list of `ints` that has length `== 3`

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    out_range: The range of output

    Returns
    -------
    None
    """
    _, _, fmap_d, fmap_h, fmap_w = shape_fm
    filter_n, _, filter_d, filter_h, filter_w = shape_filter

    pad_d = [pads[0], pads[1]]
    pad_h = [pads[2], pads[3]]
    pad_w = [pads[4], pads[5]]

    if -1 not in pad_d:
        _check_d_dimension(fmap_d, filter_d, pad_d, dilation_dhw[0])
    if -1 not in pad_h:
        _check_h_dimension(fmap_h, filter_h, pad_h, dilation_dhw[1])
    if -1 not in pad_w:
        _check_w_dimension(fmap_w, filter_w, pad_w, dilation_dhw[2])

    # C dimension should align 16
    block_size_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    block_size_m = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][0]

    block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    filter_n = ((filter_n + block_size_n - 1) //
                block_size_n) * block_size_n

    # check for not bigger than L1
    l1_buffer_size = tbe_platform.get_soc_spec("L1_SIZE")
    m_bit_ratio = {"float16": 2, "int8": 1}
    point_per_w = out_range[-1][1]
    w_in = block_size_m // point_per_w + 2
    tmp = ((w_in - 1) * stride_dhw[1] + filter_h) * fmap_w
    max_feature_map_l1 = block_size_k * tmp * m_bit_ratio[w_dtype]

    if max_feature_map_l1 > l1_buffer_size:
        error_manager_cube.raise_err_specific_user("conv3d",
            "Input is too large, the minimum tiling may exceed L1_Buffer")


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


def _pos_from_format(ele_format):
    """
    get value from ele_format
    """

    pos_n = ele_format.find('N')
    pos_c = ele_format.find('C')
    pos_d = ele_format.find('D')
    pos_h = ele_format.find('H')
    pos_w = ele_format.find('W')
    return pos_n, pos_c, pos_d, pos_h, pos_w


def _get_shape_ncdhw(shape_in, format_in):
    pos_n, pos_c, pos_d, pos_h, pos_w = _pos_from_format(format_in)
    return [shape_in[pos_n], shape_in[pos_c], shape_in[pos_d], shape_in[pos_h], shape_in[pos_w]]


def _get_attrs(strides, dilations, data_format):
    _, _, pos_d, pos_h, pos_w = _pos_from_format(data_format)
    dilation_dhw = [dilations[pos_d], dilations[pos_h], dilations[pos_w]]
    stride_dhw = [strides[pos_d], strides[pos_h], strides[pos_w]]

    return stride_dhw, dilation_dhw


def _format_normalize(fmp_format, w_format, fmp_shape, w_shape, strides,
                      dilations, groups):
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

    groups: Int of blocked connections from input channels to output channels
        Default value is 1

    Returns
    -------
    shape_fm, shape_filter, stride_dhw, dilation_dhw
    """

    shape_filter = _get_shape_ncdhw(w_shape, w_format)
    stride_dhw, dilation_dhw = _get_attrs(strides, dilations, fmp_format)
    if list(fmp_shape) == DYNAMIC_RANK_FLAG:
        shape_fm = [-1, shape_filter[1] * groups, -1, -1, -1]
    else:
        shape_fm = _get_shape_ncdhw(fmp_shape, fmp_format)

    return shape_fm, shape_filter, stride_dhw, dilation_dhw


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
    para_check.check_dtype_rule(fmp_dtype, VALID_DTYPE, "fmap")
    para_check.check_dtype_rule(w_dtype, VALID_DTYPE, "filter")
    para_check.check_dtype_rule(res_dtype, ("float16", "float32"), "output")


def _get_fmap_range(in_range, in_shape, in_format):
    if len(in_range) == FORMAT_5D_DIMS:
        # convert NCDHW/NDHWC to NDCHW
        pos_n, pos_c, pos_d, pos_h, pos_w = _pos_from_format(in_format)
        fmap_range = [in_range[pos_n], in_range[pos_d], in_range[pos_c],
                    in_range[pos_h], in_range[pos_w]]
    elif len(in_range) == FORMAT_6D_DIMS:
        # convert NDC1HWC0 to NDCHW
        fmap_range = [in_range[N_DIM_6D], in_range[D_DIM_6D], (in_shape[1], in_shape[1]),
                      in_range[H_DIM_6D], in_range[W_DIM_6D]]
    else:
        error_manager_cube.raise_err_equal_invalid('conv3d', 'range_format', 'in_format')

    return [tuple(r) for r in fmap_range]


def _get_output(x_in, k_size, pads, stride):
    return (x_in + pads[0] + pads[1] - k_size) // stride + 1


def _get_out_range(fmap_range, w_shape, pads, strides):
    fmap_range_n, fmap_range_d, fmap_range_c, fmap_range_h, fmap_range_w = fmap_range
    w_n, _, w_d, w_h, w_w = w_shape
    correct_range_flag = False
    if -1 in pads:
        # calculate output range for pad is SAME
        y_d_lower = util_common.ceil(fmap_range_d[0], strides[0])
        if fmap_range_d[1]:
            y_d_upper = util_common.ceil(fmap_range_d[1], strides[0])
        else:
            y_d_upper = None
        y_h_lower = util_common.ceil(fmap_range_h[0], strides[1])
        if fmap_range_h[1]:
            y_h_upper = util_common.ceil(fmap_range_h[1], strides[1])
        else:
            y_h_upper = HW_MAX
        y_w_lower = util_common.ceil(fmap_range_w[0], strides[2])
        # the lower limit of w_out is 2
        if y_w_lower < 2:
            lower_new = strides[2] + 1
            fmap_range_w_lower = min(lower_new, fmap_range_w[1]) if fmap_range_w[1] else lower_new
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            y_w_lower = util_common.ceil(fmap_range_w[0], strides[2])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input w \
                range is less than 2, and the lower limit of the input w range is corrected \
                as {}".format(fmap_range_w_lower))
        if fmap_range_w[1]:
            y_w_upper = util_common.ceil(fmap_range_w[1], strides[2])
        else:
            y_w_upper = HW_MAX
        pad_check_load2d_flag = True
    else:
        # calcaulate output range for pad is list
        y_d_lower = _get_output(fmap_range_d[0], w_d, (pads[0], pads[1]), strides[0])
        if y_d_lower < 1:
            fmap_range_d_lower = min(w_d, fmap_range_d[1]) if fmap_range_d[1] else w_d
            fmap_range_d = (fmap_range_d_lower, fmap_range_d[1])
            y_d_lower = _get_output(fmap_range_d[0], w_d, (pads[0], pads[1]), strides[0])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input d \
                range is less than 1, and the lower limit of the input d range is corrected \
                as {}".format(fmap_range_d_lower))
        if fmap_range_d[1]:
            y_d_upper = _get_output(fmap_range_d[1], w_d, (pads[0], pads[1]), strides[0])
        else:
            y_d_upper = None
        y_h_lower = _get_output(fmap_range_h[0], w_h, (pads[2], pads[3]), strides[1])
        if y_h_lower < 1:
            fmap_range_h_lower = min(w_h, fmap_range_h[1]) if fmap_range_h[1] else w_h
            fmap_range_h = (fmap_range_h_lower, fmap_range_h[1])
            y_h_lower = _get_output(fmap_range_h[0], w_h, (pads[2], pads[3]), strides[1])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input h \
                range is less than 1, and the lower limit of the input h range is corrected \
                as {}".format(fmap_range_h_lower))
        if fmap_range_h[1]:
            y_h_upper = _get_output(fmap_range_h[1], w_h, (pads[2], pads[3]), strides[1])
        else:
            y_h_upper = HW_MAX
        y_w_lower = _get_output(fmap_range_w[0], w_w, (pads[4], pads[5]), strides[2])
        if y_w_lower < 2:
            lower_new = w_w + strides[2]
            fmap_range_w_lower = min(lower_new, fmap_range_w[1]) if fmap_range_w[1] else lower_new
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            y_w_lower = _get_output(fmap_range_w[0], w_w, (pads[4], pads[5]), strides[2])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input w \
                range is less than 2, and the lower limit of the input w range is corrected \
                as {}".format(fmap_range_w_lower))
        if fmap_range_w[1]:
            y_w_upper = _get_output(fmap_range_w[1], w_w, (pads[4], pads[5]), strides[2])
        else:
            y_w_upper = HW_MAX
        pad_check_load2d_flag = (sum(pads) == 0)

    if y_d_lower < 1:
        error_manager_cube.raise_err_one_para('E62006', 'conv3d', 'd_out must >= 1')

    load2d_pass_flag =  ((w_d == 1) and (w_h == 1) and (w_w == 1) and
                        pad_check_load2d_flag and
                        (list(strides) == [1, 1, 1]))
    #  Chip Design demand only h_dimesion constraint
    only_fhkh_pass_flag = ((1 <= w_h <= 11) and
                           (strides[1] == 1) and
                           (y_h_lower == 1) and (y_h_upper == 1))

    #  Chip Design demand both h_dimesion and w_dimension constraint
    fhkh_fwkw_pass_flag = ((1 <= w_w <= 11) and (1 <= w_h <= 11) and
                           (strides[1] == 1) and (strides[2] == 1) and
                           (y_h_lower == 1) and (y_w_lower == 1) and
                           (y_h_upper == 1) and (y_w_upper == 1))

    if load2d_pass_flag or only_fhkh_pass_flag or fhkh_fwkw_pass_flag:
        pass
    else:
        if y_w_lower < 2:
            error_manager_cube.raise_err_one_para('E62006', 'conv3d', 'Chip Design demand w_out must >=2')

        if y_h_lower < 1:
            error_manager_cube.raise_err_one_para('E62006', 'conv3d', 'Chip Design demand h_out must >=1')
    out_range = [fmap_range[0], (y_d_lower, y_d_upper), (w_n, w_n),
                 (y_h_lower, y_h_upper), (y_w_lower, y_w_upper)]
    fmap_range = [fmap_range_n, fmap_range_d, fmap_range_c,
                  fmap_range_h, fmap_range_w]

    return out_range, fmap_range, correct_range_flag


def _check_const_dim(dim_value, dim_name):
    if not isinstance(dim_value, int):
        error_manager_cube.raise_err_specific_user("conv3d",
                 "the value of the {} dimension of shape must be int".format(dim_name))
    if dim_value <= 0:
        error_manager_cube.raise_err_specific_user("conv3d",
                 "the value of the {} dimension of shape must be -1 or >0".format(dim_name))


def _check_dynamic_mode(in_shape, w_shape, groups):
    """
    check dynamic mode
    """

    # in_shape format is NCDHW
    c_dim = 1
    fmap_dim_name_lis = ["N", "C", "D", "H", "W"]
    if DYNAMIC_FLAG not in in_shape:
        error_manager_cube.raise_err_specific_user(
            "conv3d", "need at least one dimension is a variable.")
    if DYNAMIC_FLAG in w_shape:
        error_manager_cube.raise_err_specific_user(
            "conv3d", "dynamic weight is not supported yet.")
    if in_shape[c_dim] == DYNAMIC_FLAG:
        in_shape[c_dim] = w_shape[c_dim] * groups
    for index, dim in enumerate(in_shape):
        if dim != DYNAMIC_FLAG:
            _check_const_dim(dim, fmap_dim_name_lis[index])


def _check_variable_range(range_i, mini, maxi=MAX_SHAPE_NUM, name=None):
    """
    check variable range

    """
    if not isinstance(range_i, (tuple, list)):
        error_manager_cube.raise_err_specific_user("conv3d", "type of range must be tuple or list.")
    if len(range_i) != RANGE_DIM_LEN:
        error_manager_cube.raise_err_specific_user("conv3d", "each dimension of range must be 2.")
    if not isinstance(range_i[0], int):
        error_manager_cube.raise_err_specific_user("conv3d", "The lower limit of the range must be Int.")
    if range_i[1] and (not isinstance(range_i[1], int)):
        error_manager_cube.raise_err_specific_user("conv3d", "The upper limit of the range must be Int or None.")
    if range_i[0] < mini or range_i[0] > maxi:
        error_manager_cube.raise_err_attr_range_invalid(
            "conv3d", [mini, maxi], name, range_i[0])
    if range_i[1] and (range_i[1] < mini or range_i[1] > maxi):
        error_manager_cube.raise_err_attr_range_invalid(
            "conv3d", [mini, maxi], name, range_i[1])


def _check_and_config_para(fmap,
                           weight,
                           bias,
                           offset_w,
                           output,
                           strides,
                           pads,
                           dilations,
                           groups):

    in_shape = list(fmap.get("ori_shape"))
    w_shape = list(weight.get("ori_shape"))
    in_dtype = fmap.get("dtype")
    w_dtype = weight.get("dtype")
    res_dtype = output.get("dtype")
    in_format = fmap.get("ori_format")
    w_format = weight.get("ori_format")
    in_range = fmap.get("range")

    in_dtype = in_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()

    if len(strides) != STRIDE_LENGTH:
        error_manager_cube.raise_err_specific_user("conv3d", "strides should be 5d list")
    if len(dilations) != DILATION_LENGTH:
        error_manager_cube.raise_err_specific_user("conv3d", "dilations should be 5d list")

    # check dilations for it1
    if len(set(dilations)) != 1 or dilations[2] != 1:
        error_manager_cube.raise_err_three_paras('E62001', 'Conv3D', str(dilations[2]),
            str(dilations[3]), str(dilations[1]))

    if len(pads) != PADS_LENGTH:
        error_manager_cube.raise_err_one_para('E62501', 'conv3d', 'pads')

    if list(in_shape) != DYNAMIC_RANK_FLAG:
        if len(in_shape) != SHAPE_DIMS:
            error_manager_cube.raise_err_one_para('E62501', 'conv3d', 'in_shape')

    para_check.check_shape_rule(w_shape, min_dim=SHAPE_DIMS,
                                max_dim=SHAPE_DIMS)

    if in_format not in FMAP_FORMAT_WHITE_LIST:
        error_manager_cube.raise_err_input_format_invalid(
            'conv3d', 'input', FMAP_FORMAT_WHITE_LIST, in_format)

    if w_format not in FILTER_FORMAT_WHITE_LIST:
        error_manager_cube.raise_err_input_format_invalid(
            'conv3d', 'weight', FILTER_FORMAT_WHITE_LIST, w_format)
    # shape_fm/shape_filter format is NCDHW
    shape_fm, shape_filter, stride_dhw, dilation_dhw = _format_normalize(
        in_format, w_format, in_shape, w_shape, strides, dilations, groups)

    if bias:
        util_conv3d.check_bias(bias, res_dtype)
        bias_dtype = bias.get("dtype").lower()
        para_check.check_dtype_rule(bias_dtype, ("float16", "float32"), "bias")

    if offset_w:
        error_manager_cube.raise_err_specific_user(
            'conv3d', "offset_w is not supported yet in dynamic conv3d")

    cin0 = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    cout0 = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    _check_conv3d_dtype(in_dtype, w_dtype, res_dtype)
    _check_dynamic_mode(shape_fm, shape_filter, groups)
    # calculate fmap_range
    if list(in_shape) == DYNAMIC_RANK_FLAG:
        fmap_range = [(1, None), (1, None),
                      (shape_filter[1] * groups, shape_filter[1] * groups),
                      (1, None), (1, None)]
    else:
        fmap_range = _get_fmap_range(in_range, shape_fm, in_format)

    # check fmap_range
    _, _, _, h_range, w_range = fmap_range
    _check_variable_range(h_range, HW_MIN, HW_MAX, "fmap_h")
    _check_variable_range(w_range, HW_MIN, HW_MAX, "fmap_w")
    name_lis = ['fmap_batch', 'fmap_d', 'fmap_c']
    for index, dim_range in enumerate(fmap_range[:3]):
        _check_variable_range(dim_range, 1, name=name_lis[index])
    _common_check(shape_filter, stride_dhw)

    # calculate out_range
    out_range, fmap_range, correct_range_flag = _get_out_range(fmap_range, shape_filter, pads,
                                                               stride_dhw)
    _check_groups_validation(shape_fm[1], shape_filter[1], groups)
    # calculate group parameter
    group_dict = util_common.calculate_group(shape_fm[1], shape_filter[0],
                                             groups, cout0, cin0)
    # C dimension 16 aligned
    _check_conv3d_shape(shape_fm, shape_filter, pads,
                        stride_dhw, dilation_dhw, in_dtype,
                        w_dtype, out_range)

    config_dict = {
        "shape_fm": shape_fm,
        "shape_filter": shape_filter,
        "stride_dhw": stride_dhw,
        "dilation_dhw": dilation_dhw,
        "group_dict": group_dict,
        "in_dtype": in_dtype,
        "w_dtype": w_dtype,
        "res_dtype": res_dtype,
        "fmap_range": fmap_range,
        "out_range": out_range,
        "correct_range_flag":correct_range_flag,
    }
    return config_dict


def _calc_pads(fmap_shape_ndc1hwc0, shape_filter, stride_dhw, dilation_dhw, pads):
    """
    calculate pads
    """
    _, fmap_d, _, fmap_h, fmap_w, _ = fmap_shape_ndc1hwc0
    _, _, filter_d, filter_h, filter_w = shape_filter
    stride_d, stride_h, stride_w = stride_dhw
    dilation_d, dilation_h, dilation_w = dilation_dhw

    filter_d_dilation = (filter_d - 1) * dilation_d + 1
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    if -1 in pads:
        if list(stride_dhw) == [1, 1, 1] and [filter_d, filter_h, filter_w] == [1, 1, 1]:
            return [0, 0, 0, 0, 0, 0]
        pad_d = \
            util_common.ceil(fmap_d, stride_d) * stride_d - stride_d + filter_d_dilation - fmap_d
        pad_d = tvm.max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head
        pad_h = \
            util_common.ceil(fmap_h, stride_h) * stride_h - stride_h + filter_h_dilation - fmap_h
        pad_h = tvm.max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = \
            util_common.ceil(fmap_w, stride_w) * stride_w - stride_w + filter_w_dilation - fmap_w
        pad_w = tvm.max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right
    return pads


def _conv3d_compute(fmap,
                    weight,
                    bias,
                    offset_w,
                    output,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1, 1),
                    groups=1,
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
    tvm compute
    """

    # shape_fm/shape_filter format is NCDHW, fmap_range/out_range format is NDCHW
    config_dict = _check_and_config_para(fmap, weight, bias, offset_w, output, strides, pads, dilations, groups)
    shape_fm = config_dict.get('shape_fm')
    shape_filter = config_dict.get('shape_filter')
    stride_dhw = config_dict.get('stride_dhw')
    dilation_dhw = config_dict.get('dilation_dhw')
    group_dict = config_dict.get('group_dict')
    fmp_dtype = config_dict.get('in_dtype')
    w_dtype = config_dict.get('w_dtype')
    res_dtype = config_dict.get('res_dtype')
    fmap_range = config_dict.get('fmap_range')
    out_range = config_dict.get('out_range')
    correct_range_flag = config_dict.get('correct_range_flag')
    offset_w = None
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

    # convert fmap shape to ndc1hwc0
    batch, cin, fmp_d, fmp_h, fmp_w = shape_fm
    shape_fmp_ndc1hwc0 = [batch, fmp_d, (cin + fmp_block_k - 1) // fmp_block_k, fmp_h, fmp_w,
                          fmp_block_k]
    # convert filter shape to frac_z
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    cout_ori = group_dict["cout_ori"]
    _, _, w_d, w_h, w_w = shape_filter
    shape_w_frac_z = (real_g * w_d * cin1_g * w_h * w_w, cout_g // w_block_n,
                        w_block_n, w_block_k)
    mad_dtype = _get_mad_dtype(w_dtype)
    # define var
    if shape_fmp_ndc1hwc0[N_DIM_6D] == DYNAMIC_FLAG:
        shape_fmp_ndc1hwc0[N_DIM_6D] = operation.var("batch_n", fmap_range[N_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[N_DIM_6D])
    if shape_fmp_ndc1hwc0[D_DIM_6D] == DYNAMIC_FLAG:
        shape_fmp_ndc1hwc0[D_DIM_6D] = operation.var("fmap_d", fmap_range[D_DIM_6D])
        d_out = operation.var("d_out", out_range[D_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[D_DIM_6D])
        operation.add_exclude_bound_var(d_out)
    if shape_fmp_ndc1hwc0[H_DIM_6D] == DYNAMIC_FLAG:
        shape_fmp_ndc1hwc0[H_DIM_6D] = operation.var("fmap_h", fmap_range[H_DIM_6D])
        h_out = operation.var("h_out", out_range[H_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[H_DIM_6D])
        operation.add_exclude_bound_var(h_out)
    if shape_fmp_ndc1hwc0[W_DIM_6D] == DYNAMIC_FLAG:
        shape_fmp_ndc1hwc0[W_DIM_6D] = operation.var("fmap_w", fmap_range[W_DIM_6D])
        w_out = operation.var("w_out", out_range[W_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[W_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[W_DIM_6D])

    data = tvm.placeholder(shape_fmp_ndc1hwc0, name='Fmap', dtype=fmp_dtype)
    weight = tvm.placeholder(shape_w_frac_z, name='Filter', dtype=w_dtype)

    # calculate pads
    pads = _calc_pads(shape_fmp_ndc1hwc0, shape_filter, stride_dhw, dilation_dhw, pads)

    bias_tensor = None
    if bias is not None:
        align_mod = _ALIGN_BYTE // _DTYPE_SIZE.get(res_dtype, _DEFAULT_FP16_SIZE)
        bias_align_shape = (cout_ori + align_mod - 1) // align_mod * align_mod
        bias_tensor = tvm.placeholder((bias_align_shape,),
                                      name='bias_tensor',
                                      dtype=res_dtype)
    para_dict = {
        "dsl_flag": False,
        "bias_tensor": bias_tensor,
        "pads": pads,
        "strides": stride_dhw,
        "dilation_dhw": dilation_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "correct_range_flag":correct_range_flag,
        "groups": groups,
        "ori_tensors": {"fmap": fmap,
                        "weight": weight,
                        "bias": bias,
                        "output": output}
    }
    conv_res = tbe.conv3d(data, weight, shape_filter, para_dict)

    if bias:
        return {"op_placeholder": [data, weight, bias_tensor], "op_res": [conv_res]}
    return {"op_placeholder": [data, weight], "op_res": [conv_res]}


def _check_correct_fuzz_input_range(fmap, weight, pads, strides, dilations, groups, is_dynamic_fuzz_mode):
    in_shape = list(fmap.get("ori_shape"))
    w_shape = list(weight.get("ori_shape"))
    in_format = fmap.get("ori_format")
    w_format = weight.get("ori_format")
    in_range = fmap.get("ori_range")
    # shape_fm/shape_filter format is NCDHW
    shape_fm, shape_filter, stride_dhw, _ = _format_normalize(
        in_format, w_format, in_shape, w_shape, strides, dilations, groups)
    # fmap_range -> NDCHW
    fmap_range = _get_fmap_range(in_range, shape_fm, in_format)
    # calculate out_range
    try:
        _, correct_fmap_range, correct_range_flag = _get_out_range(fmap_range, shape_filter, pads, stride_dhw)
    except RuntimeError as exc:
        if is_dynamic_fuzz_mode:
            return LOWER_LIST
        else:
            return UNSUPPORT_LIST
    finally:
        pass
    if correct_range_flag:
        if is_dynamic_fuzz_mode:
            return LOWER_LIST
        else:
            fmap_range_n, fmap_range_d, fmap_range_c, fmap_range_h, fmap_range_w = correct_fmap_range
            if in_format == "NDHWC":
                fmap_range = [fmap_range_n, fmap_range_d, fmap_range_h, fmap_range_w, fmap_range_c]
            else:
                fmap_range = [fmap_range_n, fmap_range_c, fmap_range_d, fmap_range_h, fmap_range_w]
        fmap["ori_range"] = fmap_range
    return []


def _check_l1_size(fmap, weight, pads, strides, dilations, is_dynamic_fuzz_mode):
    """
    check exceed l1 buf
    graph mode fuzz, check range[high]
    single mode fuzz, check shape and modify range[high]
    """
    def _get_l1_size(w_in):
        if DYNAMIC_FLAG in pads:
            w_out = w_in + stride_w - 1 // stride_w
        else:
            w_out = (w_in + (pad_left + pad_right) - filter_dilated_w) // stride_w + 1
        limit_h_out = math.floor(block_size_m / w_out) + _DEFAULT_L1_HO_LEN
        hw_size = ((limit_h_out - 1) * stride_h + filter_dilated_h) * w_in
        return hw_size * block_size_k * _DTYPE_SIZE.get(fmap_dtype, _DEFAULT_FP16_SIZE)

    l1_buffer_size = tbe_platform.get_soc_spec("L1_SIZE")
    fmap_dtype = fmap["dtype"]
    block_size_k = tbe_platform.CUBE_MKN[fmap_dtype]['mac'][1]
    block_size_m = tbe_platform.CUBE_MKN[fmap_dtype]['mac'][0]
    idx_w = fmap.get("ori_format").find("W")
    idx_h = fmap.get("ori_format").find("H")
    stride_h = strides[idx_h]
    stride_w = strides[idx_w]
    _, _, _, _, pad_left, pad_right = pads
    dilation_w = dilations[idx_w]
    dilation_h = dilations[idx_h]
    filter_w = weight.get("ori_shape")[weight.get("ori_format").find('W')]
    filter_h = weight.get("ori_shape")[weight.get("ori_format").find('H')]
    filter_dilated_w = (filter_w - 1) * dilation_w + 1
    filter_dilated_h = (filter_h - 1) * dilation_h + 1
    if is_dynamic_fuzz_mode:
        w_in = fmap.get("ori_range")[idx_w][0]
        limit_size = _get_l1_size(w_in)
        if limit_size > l1_buffer_size:
            return LOWER_LIST
        w_in = fmap.get("ori_range")[idx_w][1]
        limit_size = _get_l1_size(w_in)
        if limit_size > l1_buffer_size:
            return UPPER_LIST
    else:
        w_in = fmap.get("ori_shape")[idx_w]
        limit_size = _get_l1_size(w_in)
        if limit_size > l1_buffer_size:
            return UNSUPPORT_LIST
    if not is_dynamic_fuzz_mode:
        fmap_w_min = fmap.get("ori_range")[idx_w][0]
        fmap_w_max = fmap.get("ori_range")[idx_w][1]
        w_left = fmap_w_min
        w_right = fmap_w_max
        tmp_w = fmap_w_max
        while (w_right - w_left) != 1:
            max_fmap_l1 = _get_l1_size(tmp_w)
            if max_fmap_l1 > l1_buffer_size:
                w_right = tmp_w
            else:
                w_left = tmp_w
            tmp_w = w_left + (w_right - w_left) // _BINARY_SEARCH_NUM
            if w_left == fmap_w_max:
                break
        if w_in > w_left:
            fmap.get("ori_range")[idx_w] = (w_in, w_in)
        else:
            fmap.get("ori_range")[idx_w] = (fmap_w_min, w_left)
    return []


@tbe_register.register_param_generalization("Conv3D")
def conv3d_generalization(fmap,
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
                          generalize_config=None):
    """
    algorithm: conv3d_generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

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

    generalize_config: dict
    support keep_rank

    Returns
    -------
    list of params list
    """
    support_mode = ["keep_rank"]
    is_generalize_config = (generalize_config is not None and generalize_config.get("mode") in support_mode)
    if not is_generalize_config:
        return
    result = []
    is_dynamic_fuzz_mode = util_conv3d.check_fuzz_dynamic_mode(fmap)
    check_result = util_conv3d.check_para_fuzz_compile_3d(fmap, output, weight, dilations, strides, pads,
                                                          is_dynamic_fuzz_mode, _OP_TYPE, 0)
    if check_result:
        return check_result
    fmap = util_cube_dynamic.gen_conv_shape_range(fmap, _OP_TYPE, is_dynamic_fuzz_mode)
    util_conv3d.get_range(fmap)
    # check output_d and output_h and output_w
    new_pads = util_conv3d.correct_pads(fmap, output, weight, strides, pads, is_dynamic_fuzz_mode)
    err_json = _check_correct_fuzz_input_range(fmap, weight, new_pads, strides, dilations, groups, is_dynamic_fuzz_mode)
    if err_json:
        return err_json
    is_exceed_l1_lst = _check_l1_size(fmap, weight, new_pads, strides, dilations, is_dynamic_fuzz_mode)
    if is_exceed_l1_lst:
        warnings.warn("Conv3d generalization fuzz build exceed l1 buffer size.")
        return is_exceed_l1_lst
    if not is_dynamic_fuzz_mode:
        util_conv3d.generalize_input_keep_rank(fmap)
        util_conv3d.generalize_input_keep_rank(output)
    try:
        _check_and_config_para(fmap, weight, bias, offset_w, output, strides, new_pads, dilations, groups)
    except RuntimeError as exc:
        return UNSUPPORT_LIST
    finally:
        pass
    result.append([fmap, weight, bias, offset_w, output, {"strides": strides},
                   {"pads": pads}, {"dilations": dilations}, {"groups": groups},
                   {"data_format": data_format}, {"offset_x": offset_x}])
    return result


@register_operator("Conv3D")
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

    with tbe.compute():
        res = _conv3d_compute(
            fmap, weight, bias, offset_w, output, strides, pads, dilations,
            groups, offset_x, kernel_name)

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
