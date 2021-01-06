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
from te import tvm
import te.lang.cce as tbe
from te.lang.cce.te_compute import conv3d_compute
import te.lang.dynamic as dynamic
import te.platform as tbe_platform
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils.error_manager import error_manager_util
from impl.util import util_common

BIAS_LENGTH = 1
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


def _common_check(shape_filter, stride_dhw):
    _, _, filter_d, filter_h, filter_w = shape_filter
    stride_d, stride_h, stride_w = stride_dhw
    if filter_d < FILTER_DHW_MIN or filter_d > FILTER_DHW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'weight',
            'dim': 'D',
            'range': '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX),
            'actual_value': str(filter_d)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_d < STRIDE_MIN or stride_d > STRIDE_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'stride',
            'dim': 'D',
            'range': '[{}, {}]'.format(STRIDE_MIN, STRIDE_MAX),
            'actual_value': str(stride_d),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if filter_h < FILTER_DHW_MIN or filter_h > FILTER_DHW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'filter',
            'dim': 'H',
            'range': '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX),
            'actual_value': str(filter_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_h < STRIDE_MIN or stride_h > STRIDE_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'stride',
            'dim': 'H',
            'range': '[{}, {}]'.format(STRIDE_MIN, STRIDE_MAX),
            'actual_value': 'stride_h = {}'.format(stride_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if filter_w < FILTER_DHW_MIN or filter_w > FILTER_DHW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'filter',
            'dim': 'W',
            'range': '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX),
            'actual_value': str(filter_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_w < STRIDE_MIN or stride_w > STRIDE_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'stride',
            'dim': 'W',
            'range': '[{}, {}]'.format(STRIDE_MIN, STRIDE_MAX),
            'actual_value': str(stride_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_d_dimension(fmap_d, filter_d, pad_d, stride_d, dilation_d):
    if (fmap_d + pad_d[0] + pad_d[1]) < ((filter_d - 1) * dilation_d + 1):
        dict_args = {
            'errCode': 'E60012',
            'depth_of_x': str(fmap_d + pad_d[0] + pad_d[1]),
            'depth_of_filter': str((filter_d - 1) * dilation_d - 1),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_d[0] < PAD_MIN or pad_d[1] < PAD_MIN or pad_d[0] > PAD_MAX or pad_d[1] > PAD_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'pad',
            'dim': 'D',
            'range': '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'actual_value': 'pad_d[0] = {}, pad_d[1] = {}'.format(pad_d[0],
                                                                  pad_d[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_d[0] >= filter_d or pad_d[1] >= filter_d:
        dict_args = {
            'errCode': 'E60013',
            'depth_of_pad': 'pad_d[0] = {}, pad_d[1] = {}'.format(pad_d[0],
                                                                  pad_d[1]),
            'depth_of_filter': str(filter_d)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_h_dimension(fmap_h, filter_h, pad_h, stride_h, dilation_h):
    if fmap_h < FMAP_HW_MIN or fmap_h > FMAP_HW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'input',
            'dim': 'H',
            'range': '[{}, {}]'.format(FMAP_HW_MIN, FMAP_HW_MAX),
            'actual_value': str(fmap_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_h[0] < PAD_MIN or pad_h[1] < PAD_MIN or pad_h[0] > PAD_MAX or pad_h[1] > PAD_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'pad',
            'dim': 'H',
            'range': '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'actual_value': 'pad_h[0] = {}, pad_h[1] = {}'.format(pad_h[0],
                                                                  pad_h[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if (fmap_h + pad_h[0] + pad_h[1]) < ((filter_h - 1) * dilation_h + 1):
        # Chip Design demand, Load3D
        dict_args = {
            'errCode': 'E60014',
            'h_of_x': str(fmap_h + pad_h[0] + pad_h[1]),
            'h_of_filter': str(filter_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_h[0] >= filter_h or pad_h[1] >= filter_h:
        dict_args = {
            'errCode': 'E60016',
            'h_of_filter': str(filter_h),
            'h_of_pad': '[pad_h[0]={}, pad_h[1]={}]'.format(pad_h[0], pad_h[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_w_dimension(fmap_w, filter_w, pad_w, stride_w, dilation_w):
    if fmap_w < FMAP_HW_MIN or fmap_w > FMAP_HW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'input',
            'dim': 'W',
            'range': '[{}, {}]'.format(FMAP_HW_MIN, FMAP_HW_MAX),
            'actual_value': str(fmap_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_w[0] < PAD_MIN or pad_w[1] < PAD_MIN or pad_w[0] > PAD_MAX or pad_w[1] > PAD_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'pad',
            'dim': 'W',
            'range': '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'actual_value': 'pad_w[0] = {}, pad_w[1] = {}'
                            .format(pad_w[0], pad_w[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if filter_w > (fmap_w + pad_w[0] + pad_w[1]):
        # Chip Design demand, Load3D
        dict_args = {
            'errCode': 'E60015',
            'w_of_x': str(fmap_w + pad_w[0] + pad_w[1]),
            'w_of_filter': str(filter_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_w[0] >= filter_w or pad_w[1] >= filter_w:
        dict_args = {
            'errCode': 'E60017',
            'w_of_filter': str(filter_w),
            'w_of_pad': '[pad_w[0]={}, pad_w[1]={}]'.format(pad_w[0], pad_w[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_conv3d_shape(shape_fm, shape_filter, pads, stride_dhw, dilation_dhw,
                       fmp_dtype, w_dtype, groups, dynamic_mode=None,
                       fmap_range=None, out_range=None):
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

    groups: The groups for group convolution

    dynamic_mode: The mode of dynamic shape

    fmap_range: The range of feature map

    out_range: The range of output

    Returns
    -------
    None
    """
    if shape_fm[1] != shape_filter[1] * groups:
        dict_args = {
            'errCode': 'E60010',
            'channel_of_x': str(shape_fm[1]),
            'channel_of_filter': str(shape_filter[1] * groups)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    fmap_n, fmap_c, fmap_d, fmap_h, fmap_w = shape_fm
    filter_n, filter_c, filter_d, filter_h, filter_w = shape_filter

    pad_d = [pads[0], pads[1]]
    pad_h = [pads[2], pads[3]]
    pad_w = [pads[4], pads[5]]

    if dynamic_mode != "dynamic_dhw":
        _check_d_dimension(fmap_d, filter_d, pad_d, stride_dhw[0], dilation_dhw[0])
        _check_h_dimension(fmap_h, filter_h, pad_h, stride_dhw[1], dilation_dhw[1])
        _check_w_dimension(fmap_w, filter_w, pad_w, stride_dhw[2], dilation_dhw[2])

    # C dimension should align 16
    block_size_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    block_size_m = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][0]
    famp_c = ((fmap_c + block_size_k - 1) //
              block_size_k) * block_size_k
    filter_c = fmap_c
    block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    filter_n = ((filter_n + block_size_n - 1) //
                block_size_n) * block_size_n

    # calculated by h_i and w_i
    h_out = (fmap_h + (pad_h[0] + pad_h[1]) - filter_h) // stride_dhw[1] + 1
    w_out = (fmap_w + (pad_w[0] + pad_w[1]) - filter_w) // stride_dhw[2] + 1
    d_out = (fmap_d + (pad_d[0] + pad_d[1]) - filter_d) // stride_dhw[0] + 1

    load2d_pass_flag = ((filter_d == 1) and (filter_h == 1) and (filter_w == 1) and
                        (list(pads) == [0, 0, 0, 0, 0, 0]) and
                        (list(stride_dhw) == [1, 1, 1]))

    #  Chip Design demand only h_dimesion constraint
    only_fhkh_pass_flag = ((1 <= filter_h <= 11) and
                           (stride_dhw[1] == 1) and
                           (h_out == 1))

    #  Chip Design demand both h_dimesion and w_dimension constraint
    fhkh_fwkw_pass_flag = ((1 <= filter_w <= 11) and (1 <= filter_h <= 11) and
                           (stride_dhw[1] == 1) and (stride_dhw[2] == 1) and
                           (h_out == 1) and (w_out == 1))

    if load2d_pass_flag or only_fhkh_pass_flag or fhkh_fwkw_pass_flag or dynamic_mode == "dynamic_dhw":
        pass
    else:
        if w_out < 2:
            # Chip Design demand w_out must >=2
            dict_args = {
                'errCode': 'E62006',
                'error_desc': 'Chip Design demand w_out must >=2'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

        if h_out < 2:
            # Chip Design demand h_out must >=2
            dict_args = {
                'errCode': 'E62006',
                'error_desc': 'Chip Design demand h_out must >=2'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    # check for not bigger than L1
    l1_buffer_size = tbe_platform.get_soc_spec("L1_SIZE")
    m_bit_ratio = {"float16": 2, "int8": 1}
    if dynamic_mode == "dynamic_dhw":
        point_per_w = out_range[-1][1]
    else:
        point_per_w = (fmap_w - filter_w +
                       pad_w[0] + pad_w[1]) // stride_dhw[2] + 1
    w_in = block_size_m // point_per_w + 2
    tmp = ((w_in - 1) * stride_dhw[1] + filter_h) * fmap_w
    max_feature_map_l1 = block_size_k * tmp * m_bit_ratio[w_dtype]

    if max_feature_map_l1 > l1_buffer_size:
        dict_args = {
            'errCode': 'E60026',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


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

    pos_n, pos_c, pos_d, pos_h, pos_w = _pos_from_format(fmp_format)
    shape_fm = _get_shape_ncdhw(fmp_shape, fmp_format)
    shape_filter = _get_shape_ncdhw(w_shape, w_format)
    stride_dhw, dilation_dhw = _get_attrs(strides, dilations, fmp_format)

    return shape_fm, shape_filter, stride_dhw, dilation_dhw


def _ceil(x_1, x_2):
    if x_2 == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return (x_1 + x_2 - 1) // x_2


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
        dict_args = {
            'errCode': 'E61301',
            'param_name_1': 'range_format',
            'param_name_2': 'in_format',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return [tuple(r) for r in fmap_range]


def _get_output(x_in, k_size, pads, stride):
    return (x_in + pads[0] + pads[1] - k_size) // stride + 1


def _get_out_range(fmap_range, w_shape, pads, strides, dilations):
    fmap_range_n, fmap_range_d, fmap_range_c, fmap_range_h, fmap_range_w = fmap_range
    w_n, w_c, w_d, w_h, w_w = w_shape

    if -1 in pads:
        # calculate output range for pad is SAME
        y_d_lower = _ceil(fmap_range_d[0], strides[0])
        y_d_upper = _ceil(fmap_range_d[1], strides[0])
        y_h_lower = _ceil(fmap_range_h[0], strides[1])
        y_h_upper = _ceil(fmap_range_h[1], strides[1])
        y_w_lower = _ceil(fmap_range_w[0], strides[2])
        y_w_upper = _ceil(fmap_range_w[1], strides[2])
        pad_check_load2d_flag = True
    else:
        # calcaulate output range for pad is list
        y_d_lower = _get_output(fmap_range_d[0], w_d, (pads[0], pads[1]), strides[0])
        y_d_upper = _get_output(fmap_range_d[1], w_d, (pads[0], pads[1]), strides[0])
        y_h_lower = _get_output(fmap_range_h[0], w_h, (pads[2], pads[3]), strides[1])
        y_h_upper = _get_output(fmap_range_h[1], w_h, (pads[2], pads[3]), strides[1])
        y_w_lower = _get_output(fmap_range_w[0], w_w, (pads[4], pads[5]), strides[2])
        y_w_upper = _get_output(fmap_range_w[1], w_w, (pads[4], pads[5]), strides[2])
        pad_check_load2d_flag = True if sum(pads) == 0 else False

    if y_d_lower < 1:
        dict_args = {
            'errCode': 'E62006',
            'error_desc': 'd_out must >= 1'
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

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
            # Chip Design demand w_out must >=2
            dict_args = {
                'errCode': 'E62006',
                'error_desc': 'Chip Design demand w_out must >=2'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

        if y_h_lower < 2:
            # Chip Design demand h_out must >=2
            dict_args = {
                'errCode': 'E62006',
                'error_desc': 'Chip Design demand h_out must >=2'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    return [fmap_range[0], (y_d_lower, y_d_upper), (w_n,w_n),
            (y_h_lower, y_h_upper), (y_w_lower, y_w_upper)]


def _check_const_dim(dim_value):
    if type(dim_value) != int:
        args_dict = {
            "errCode": "E60037",
            "param_name": "shape axis",
            "type_list": "[int]",
            "type": "{}".format(type(dim_value))
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    if dim_value <= 0:
        args_dict = {
            "errCode": "E60039",
            "attr_name": "axis",
            "param_name": "shape",
            "comparator": "more",
            "expected_value": "0",
            "input_value": "{}".format(dim_value)
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )


def _check_dynamic_mode(in_shape, w_shape):
    """
    config dynamic mode
    """

    # in_shape format is NCDHW
    n_dim = 0
    c_dim = 1
    d_dim = 2
    h_dim = 3
    w_dim = 4
    dynamic_mode = None
    if in_shape[h_dim] == -1 and in_shape[w_dim] == -1 \
            and in_shape[d_dim] == -1 and in_shape[n_dim] != -1 \
            and in_shape[c_dim] != -1 and -1 not in w_shape:
        dynamic_mode = "dynamic_dhw"
        _check_const_dim(in_shape[n_dim])
        _check_const_dim(in_shape[c_dim])
    elif in_shape[n_dim] == -1 and in_shape[c_dim] != -1 and in_shape[h_dim] != -1 \
            and in_shape[w_dim] != -1 and in_shape[d_dim] != -1 and -1 not in w_shape:
        dynamic_mode = "dynamic_batch"
        _check_const_dim(in_shape[c_dim])
        _check_const_dim(in_shape[h_dim])
        _check_const_dim(in_shape[w_dim])
        _check_const_dim(in_shape[d_dim])
    else:
        dict_args = {
            'errCode': 'E50060',
            'op_name': 'dynamic conv3d',
            'description': 'dynamic only support dynamic dhw or dynamic batch',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    return dynamic_mode


def _check_variable_range(variable, mini, maxi=MAX_SHAPE_NUM, name=None):
    """
    check variable range

    """
    if (not isinstance(variable, int)) or variable < mini or variable > maxi:
        dict_args = dict()
        dict_args["errCode"] = "E65006"
        dict_args["op_name"] = 'dynamic conv3d'
        dict_args["range"] = "[{},{}]".format(mini, maxi)
        dict_args["attr_name"] = name
        dict_args["value"] = str(variable)
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_and_config_para(fmap,
                           weight,
                           bias,
                           offset_w,
                           output,
                           strides,
                           pads,
                           dilations,
                           groups,
                           data_format,
                           offset_x,
                           kernel_name):

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

    if len(in_shape) != SHAPE_DIMS:
        dict_args = {
            'errCode': 'E62501',
            'param_name': 'in_shape',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    para_check.check_shape_rule(w_shape, min_dim=SHAPE_DIMS,
                                max_dim=SHAPE_DIMS)

    if in_format not in ['NCDHW', 'NDHWC']:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'input',
            'expected_format_list': '[{}, {}]'.format('NCDHW', 'NDHWC'),
            'format': in_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    if w_format not in ['NCDHW', 'NDHWC', 'DHWCN']:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': '[{}, {}, {}]'.format('NCDHW', 'NDHWC', 'DHWCN'),
            'format': w_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    shape_fm, shape_filter, stride_dhw, dilation_dhw = _format_normalize(
        in_format, w_format, in_shape, w_shape, strides, dilations)

    if bias:
        dict_args = {
            'errCode': 'E50060',
            'op_name': 'dynamic conv3d',
            'description': "bias is not supported yet in dynamic conv3d"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if offset_w:
        dict_args = {
            'errCode': 'E50060',
            'op_name': 'dynamic conv3d',
            'description': "offset_w is not supported yet in dynamic conv3d"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if groups != 1:
        dict_args = {
            'errCode': 'E50060',
            'op_name': 'dynamic conv3d',
            'description': "only supports groups=1"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    cin0 = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    cout0 = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    _check_conv3d_dtype(in_dtype, w_dtype, res_dtype)
    dynamic_mode = _check_dynamic_mode(shape_fm, shape_filter)
    # calculate fmap_range
    fmap_range = _get_fmap_range(in_range, shape_fm, in_format)

    _common_check(shape_filter, stride_dhw)
    # calculate out_range
    out_range = _get_out_range(fmap_range, shape_filter, pads, stride_dhw, dilation_dhw)
    # calculate group parameter
    group_dict = util_common.calculate_group(shape_fm[1], shape_filter[0],
                                             groups, cout0, cin0)
    # C dimension 16 aligned
    _check_conv3d_shape(shape_fm, shape_filter, pads,
                        stride_dhw, dilation_dhw, in_dtype,
                        w_dtype, groups, dynamic_mode, fmap_range,
                        out_range)
    batch_range, d_range, _, h_range, w_range = fmap_range
    _, do_range, _, ho_range, wo_range = out_range
    _check_variable_range(h_range[0], HW_MIN, HW_MAX, "fmap_h")
    _check_variable_range(h_range[1], HW_MIN, HW_MAX, "fmap_h")
    _check_variable_range(w_range[0], HW_MIN, HW_MAX, "fmap_w")
    _check_variable_range(w_range[1], HW_MIN, HW_MAX, "fmap_w")
    name_lis = ['fmap_batch', 'fmap_d', 'fmap_c']
    for index, dim_range in enumerate(fmap_range[:3]):
        _check_variable_range(dim_range[0], 1, name=name_lis[index])
        _check_variable_range(dim_range[1], 1, name=name_lis[index])

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
            _ceil(fmap_d, stride_d) * stride_d - stride_d + filter_d_dilation - fmap_d
        pad_d = tvm.max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head
        pad_h = \
            _ceil(fmap_h, stride_h) * stride_h - stride_h + filter_h_dilation - fmap_h
        pad_h = tvm.max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = \
            _ceil(fmap_w, stride_w) * stride_w - stride_w + filter_w_dilation - fmap_w
        pad_w = tvm.max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right
    else:
        if pads[0] < PAD_MIN or pads[1] < PAD_MIN or pads[0] > PAD_MAX or pads[1] > PAD_MAX:
            dict_args = {
                'errCode': 'E62003',
                'param_name': 'pad',
                'dim': 'D',
                'range': '[{}, {}]'.format(PAD_MIN, PAD_MAX),
                'actual_value': 'pads[0] = {}, pads[1] = {}'.format(pads[0],
                                                                    pads[1])
            }
            raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

        if pads[0] >= filter_d or pads[1] >= filter_d:
            dict_args = {
                'errCode': 'E60013',
                'depth_of_pad': 'pads[0] = {}, pads[1] = {}'.format(pads[0],
                                                                    pads[1]),
                'depth_of_filter': str(filter_d)
            }
            raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

        if pads[2] < PAD_MIN or pads[3] < PAD_MIN or pads[2] > PAD_MAX or pads[3] > PAD_MAX:
            dict_args = {
                'errCode': 'E62003',
                'param_name': 'pad',
                'dim': 'H',
                'range': '[{}, {}]'.format(PAD_MIN, PAD_MAX),
                'actual_value': 'pads[2] = {}, pads[3] = {}'.format(pads[2],
                                                                    pads[3])
            }
            raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

        if pads[2] >= filter_h or pads[3] >= filter_h:
            dict_args = {
                'errCode': 'E60016',
                'h_of_filter': str(filter_h),
                'h_of_pad': '[pads[2]={}, pads[3]={}]'.format(pads[2], pads[3])
            }
            raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

        if pads[4] < PAD_MIN or pads[5] < PAD_MIN or pads[4] > PAD_MAX or pads[5] > PAD_MAX:
            dict_args = {
                'errCode': 'E62003',
                'param_name': 'pad',
                'dim': 'W',
                'range': '[{}, {}]'.format(PAD_MIN, PAD_MAX),
                'actual_value': 'pads[4] = {}, pads[5] = {}'
                                .format(pads[4], pads[5])
            }
            raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

        if pads[4] >= filter_w or pads[5] >= filter_w:
            dict_args = {
                'errCode': 'E60017',
                'w_of_filter': str(filter_w),
                'w_of_pad': '[pads[4]={}, pads[5]={}]'.format(pads[4], pads[5])
            }
            raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))
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
    tvm compute
    """

    # shape_fm/shape_filter format is NCDHW, fmap_range/out_range format is NDCHW
    config_dict= \
            _check_and_config_para(fmap, weight, bias, offset_w, output, \
            strides, pads, dilations, groups, data_format, offset_x, kernel_name)
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
    bias = None
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
    if shape_fmp_ndc1hwc0[D_DIM_6D] == -1 and shape_fmp_ndc1hwc0[H_DIM_6D] == -1 and \
        shape_fmp_ndc1hwc0[W_DIM_6D] == -1:
        shape_fmp_ndc1hwc0[D_DIM_6D] = tbe_base.var("fmap_d", fmap_range[D_DIM_6D])
        shape_fmp_ndc1hwc0[H_DIM_6D] = tbe_base.var("fmap_h", fmap_range[H_DIM_6D])
        shape_fmp_ndc1hwc0[W_DIM_6D] = tbe_base.var("fmap_w", fmap_range[W_DIM_6D])
        d_out = tbe_base.var("d_out", out_range[D_DIM_6D])
        h_out = tbe_base.var("h_out", out_range[H_DIM_6D])
        w_out = tbe_base.var("w_out", out_range[W_DIM_6D])
        tbe_base.add_exclude_bound_var(shape_fmp_ndc1hwc0[D_DIM_6D])
        tbe_base.add_exclude_bound_var(shape_fmp_ndc1hwc0[H_DIM_6D])
        tbe_base.add_exclude_bound_var(shape_fmp_ndc1hwc0[W_DIM_6D])
        tbe_base.add_exclude_bound_var(d_out)
        tbe_base.add_exclude_bound_var(h_out)
        tbe_base.add_exclude_bound_var(w_out)
    elif shape_fmp_ndc1hwc0[N_DIM_6D] == -1:
        shape_fmp_ndc1hwc0[N_DIM_6D] = tbe_base.var("batch_n", fmap_range[N_DIM_6D])
        tbe_base.add_exclude_bound_var(shape_fmp_ndc1hwc0[N_DIM_6D])

    data = tvm.placeholder(shape_fmp_ndc1hwc0, name='Fmap', dtype=fmp_dtype)
    weight = tvm.placeholder(shape_w_frac_z, name='Filter', dtype=w_dtype)

    # calculate pads
    pads = _calc_pads(shape_fmp_ndc1hwc0, shape_filter, stride_dhw, dilation_dhw, pads)

    bias_tensor = None
    para_dict = {
        "dsl_flag": False,
        "bias_tensor": bias_tensor,
        "pads": pads,
        "strides": stride_dhw,
        "dilation_dhw": dilation_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }
    conv_res = conv3d_compute.conv3d(data, weight, shape_filter, para_dict)

    return {"op_placeholder": [data, weight], "op_res": [conv_res]}


@tbe_base.register_operator("Conv3D")
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

    with tbe_base.compute():
        res = _conv3d_compute(
            fmap, weight, bias, offset_w, output, strides, pads, dilations,
            groups, data_format, offset_x, kernel_name)

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
