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
dynamic depthwise_conv2d_backprop_filter
"""
from __future__ import absolute_import

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.util_cube_dynamic import CubeParaProcess


BLOCK_SIZE = tbe_platform.BLOCK_REDUCE
DYNAMIC_FLAG = -1
UNKNOWN_RANK_SHAPE = [-2]
DIM_N_NCHW = 0
DIM_C_NCHW = 1
DIM_H_NCHW = 2
DIM_W_NCHW = 3
# shape's dim of input and output must be 4 or 5
FEATURE_MAP_DIM = [4, 5]

# shape's dim of filter must be 4
FILTER_DIM = 4

# shape's dim of strides/pads must be 4
STRIDES_DIM = 4
PADS_DIM = 4

# shape's dim of dilation must be 4
DILATION_DIM = 4

#the bytes length of serveral dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}

# the max size is 2**63-1
DATA_SIZE_MAX = 9223372036854775807

# fmapH, fmapW must be in [1, 4096]
FMAP_HW_MAX = 4096
FMAP_HW_MIN = 1

# Dedy H, W must be in [2, 4096]
DEDY_HW_MAX = 4096
DEDY_H_MIN = 1
DEDY_W_MIN = 2

# filterH, filterW must be in [1, 255]
FILTER_HW_MAX = 255
FILTER_HW_MIN = 1

# stride must be in [1, 63]
STRIDE_HW_MAX = 63
STRIDE_HW_MIN = 1

# pad must be in [0, 255]
PAD_MAX = 255
PAD_MIN = 0

# C0_SIZE
C0_SIZE = 16
ORI_SHAPE_LEN = 4
SHAPE_LEN = 5


def _ceil(x_1, x_2):
    if x_2 == 0:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "Division by zero happened when ceil_div.")
    return (x_1 + x_2 - 1) // x_2


def _align(x_1, x_2):
    return _ceil(x_1, x_2) * x_2

def _get_output(x_in, k_size, pads, stride, dilation):
        return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1


def _get_input(x_out, filter_dilation, pads, stride):
    if DYNAMIC_FLAG in pads:
        input_low = stride * (x_out - 1) + 1
        input_high = stride * x_out
    else:
        input_low = (x_out - 1) * stride + filter_dilation - pads[0] - pads[1]
        input_high = (x_out - 1) * stride + filter_dilation - pads[0] - pads[1] + stride - 1
    return input_low, input_high


def _check_attr_range_dw(name, value, attr_min=None, attr_max=None):
    if value is None:
        return
    if not isinstance(value, int):
        error_manager_cube.raise_err_one_para("E62006", "depthwise_conv2d_backprop_filter",
                                    name + " must be int.")
    if value  > attr_max or value < attr_min:
        error_manager_cube.raise_err_attr_range_invalid("depthwise_conv2d_backprop_filter",
                                              str([attr_min, attr_max]), name , str(value))


def _check_range(in_range, dedy_range, in_shape, dedy_shape):
    if len(in_range) not in FEATURE_MAP_DIM:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                       "in_range", str(FEATURE_MAP_DIM), str(len(in_range)))
    elif len(in_range) == FEATURE_MAP_DIM[1]:
        in_range = [in_range[0], (in_range[1], in_shape[1]), in_range[2], in_range[3]]

    if len(dedy_range) not in FEATURE_MAP_DIM:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                       "dedy_range", str(FEATURE_MAP_DIM), str(len(dedy_range)))
    elif len(dedy_range) == FEATURE_MAP_DIM[1]:
        dedy_range = [dedy_range[0], (dedy_shape[1], dedy_shape[1]), dedy_range[2], dedy_range[3]]
    return in_range, dedy_range


def _check_shape(shape_in, shape_dedy, shape_dedw):
    """Check input shape."""
    fmap_n, fmap_c, _, _ = shape_in
    dedy_n, dedy_c, _, _ = shape_dedy
    filter_n, filter_c, _, _ = shape_dedw

    if filter_c != 1:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                       "filter_c", "1", str(filter_c))
    if fmap_c * filter_c != dedy_c:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "fmap_c*filter_c must be equal with dedy_c.")
    if fmap_n != dedy_n:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "fmap_n must be equal with dedy_n.")
    if fmap_c != filter_n:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "fmap_c must be equal with filter_n.")
    if DYNAMIC_FLAG in shape_dedw:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "dynamic filter_shape is not supported yet.")


def _check_data_format(data_format, expect_format_list):
    """
    check data format
    """
    if data_format not in expect_format_list:
        error_manager_cube.raise_err_input_params_not_expected("depthwise_conv2d_backprop_filter",
                                                     "data_foramt", str(expect_format_list),
                                                     data_format)


def _check_dilations(dilations, dim_n, dim_c, dim_h, dim_w):
    """
    check dilations dimensions
    """
    if dilations[dim_n] != 1 or dilations[dim_c] != 1 or \
        dilations[dim_h] != 1 or dilations[dim_w] != 1:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "dilations only support [1, 1, 1, 1].")


def _check_stride(strides, dim_n, dim_c):
    """
    check stride type and dim
    """
    if strides[dim_n] != 1 or strides[dim_c] != 1:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "stride only support 1 on N axis and C axis.")


def _get_dynamic_shape(fmap, dedy, dedw, fmap_range, dedy_range, strides, pads, dilations):

    fmap_n, fmap_c, fmap_h, fmap_w = fmap
    dedy_n, dedy_c, dedy_h, dedy_w = dedy
    dedw_n, dedw_c, kernel_h, kernel_w = dedw
    if fmap_n != DYNAMIC_FLAG and fmap_h != DYNAMIC_FLAG and fmap_w != DYNAMIC_FLAG:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "no dynamic shape found in fmap.")
    if fmap_n * dedy_n < 0:
        fmap_n = fmap_n if fmap_n > 0 else dedy_n
        dedy_n = fmap_n
    if fmap_h * dedy_h < 0:
        if dedy_h == DYNAMIC_FLAG:
            calculated_dedy_h = _ceil(fmap_h, strides[0])
            if pads[0] != DYNAMIC_FLAG and pads[1] != DYNAMIC_FLAG:
                calculated_dedy_h = _get_output(fmap_h, kernel_h, [pads[0], pads[1]], strides[0], dilations[2])
            dedy_h = calculated_dedy_h
    if fmap_w * dedy_w < 0:
        if dedy_w == DYNAMIC_FLAG:
            calculated_dedy_w = _ceil(fmap_w, strides[1])
            if pads[2] != DYNAMIC_FLAG and pads[3] != DYNAMIC_FLAG:
                calculated_dedy_w = _get_output(fmap_w, kernel_w, [pads[2], pads[3]], strides[1], dilations[3])
            dedy_w = calculated_dedy_w

    if fmap_c == DYNAMIC_FLAG:
        fmap_c = dedw_n
        dedy_c = fmap_c * dedw_c
    if fmap_n == DYNAMIC_FLAG:
        fmap_n = operation.var("batch", bound=fmap_range[0])
        dedy_n = fmap_n
        operation.add_exclude_bound_var(fmap_n)
    if fmap_h == DYNAMIC_FLAG:
        fmap_h = operation.var("fmap_h", bound=fmap_range[2])
        dedy_h = operation.var("dedy_h", bound=dedy_range[2])
        operation.add_exclude_bound_var(fmap_h)
        operation.add_exclude_bound_var(dedy_h)
    if fmap_w == DYNAMIC_FLAG:
        fmap_w = operation.var("fmap_w", bound=fmap_range[3])
        dedy_w = operation.var("dedy_w", bound=dedy_range[3])
        operation.add_exclude_bound_var(fmap_w)
        operation.add_exclude_bound_var(dedy_w)
    return (fmap_n, fmap_c, fmap_h, fmap_w), (dedy_n, dedy_c, dedy_h, dedy_w)


def  _range_correction(fmap_range, kernel, pads, stride, dilation, out_shape):
    ranges = [*fmap_range[0], *fmap_range[2], *fmap_range[3]]
    if None in ranges:
        return fmap_range
    if DYNAMIC_FLAG in pads:
        out_h_lower = _ceil(fmap_range[2][0], stride[2])
        out_h_upper = _ceil(fmap_range[2][1], stride[2])
        out_w_lower = _ceil(fmap_range[3][0], stride[3])
        out_w_upper = _ceil(fmap_range[3][1], stride[3])
    else:
        out_h_lower = _get_output(fmap_range[2][0], kernel[2], 
                                  (pads[0], pads[1]), stride[2], dilation[2])
        out_h_upper = _get_output(fmap_range[2][1], kernel[2], 
                                  (pads[0], pads[1]), stride[2], dilation[2])
        out_w_lower = _get_output(fmap_range[3][0], kernel[3], 
                                  (pads[2], pads[3]), stride[3], dilation[3])
        out_w_upper = _get_output(fmap_range[3][1], kernel[3],
                                  (pads[2], pads[3]), stride[3], dilation[3])
    return [fmap_range[0], (out_shape[1], out_shape[1]),
            (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]


def _depthwise_conv2dbp_filter_compute(input_fm, filter_size, out_backprop, filter_grad,
                                       strides, dilations, pads, data_format, kernel_name):
    shape_in = input_fm.get('ori_shape')
    shape_dedy = out_backprop.get('ori_shape')
    shape_dedw = filter_grad.get('ori_shape')
    in_dtype = input_fm.get('dtype')
    dedy_dtype = out_backprop.get('dtype')
    dedw_dtype = filter_grad.get('dtype')

    input_ori_format = input_fm.get('ori_format')
    dedy_ori_format = out_backprop.get('ori_format')
    dedw_ori_format = filter_grad.get('ori_format')
    _check_data_format(input_ori_format, ['NCHW', 'NHWC'])
    _check_data_format(dedy_ori_format, ['NCHW', 'NHWC'])
    if not dedw_ori_format:
        dedw_ori_format = "HWCN"
    else:
        _check_data_format(dedw_ori_format, ['HWCK', 'HWCN', 'NCHW', 'NHWC'])
    _check_data_format(data_format, ['NCHW', 'NHWC'])
    if input_ori_format != data_format or dedy_ori_format != data_format:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                         "input_ori_format/dedy_ori_format must be equal with data_format.")

    if dedw_ori_format in ('HWCK', 'HWCN'):
        shape_dedw = (shape_dedw[3], shape_dedw[2], shape_dedw[0], shape_dedw[1])

    # index of origin dimension
    if input_ori_format == "NCHW":
        dim_n, dim_c, dim_h, dim_w = 0, 1, 2, 3 # NCHW
    else:
        dim_n, dim_h, dim_w, dim_c = 0, 1, 2, 3
    unknown_rank_flag = False
    if list(shape_in) == UNKNOWN_RANK_SHAPE or list(shape_dedy) == UNKNOWN_RANK_SHAPE:
        unknown_rank_flag = True
        in_channel = shape_dedw[DIM_N_NCHW]
        channel_mul = shape_dedw[DIM_C_NCHW]
        out_channel = in_channel * channel_mul
        shape_in = (DYNAMIC_FLAG, in_channel, DYNAMIC_FLAG, DYNAMIC_FLAG)
        shape_dedy = (DYNAMIC_FLAG, out_channel, DYNAMIC_FLAG, DYNAMIC_FLAG)
        in_range = [(1, None), (in_channel, in_channel), (1, None), (1, None)]
        dedy_range = [(1, None), (out_channel, out_channel), (1, None), (2, None)]
    elif input_ori_format == 'NHWC':
        shape_in = [shape_in[dim_n], shape_in[dim_c], shape_in[dim_h], shape_in[dim_w]]
        shape_dedy = [shape_dedy[dim_n], shape_dedy[dim_c], shape_dedy[dim_h], shape_dedy[dim_w]]
        in_range = list(input_fm.get('range'))
        dedy_range = list(out_backprop.get('range'))
    else:
        in_range = list(input_fm.get('range'))
        dedy_range = list(out_backprop.get('range'))
    shape_in, shape_dedy = _get_dynamic_shape(shape_in, shape_dedy, shape_dedw, in_range, dedy_range, 
                                              [strides[dim_h], strides[dim_w]], pads, dilations)

    para_check.check_dtype(in_dtype.lower(), ('float16',), param_name='input_fm')
    para_check.check_dtype(dedy_dtype.lower(), ('float16',), param_name='out_backprop')
    para_check.check_dtype(dedw_dtype.lower(), ('float32',), param_name='filter_grad')

    if len(strides) != STRIDES_DIM:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                                 "strides", str(STRIDES_DIM), str(len(strides)))
    if len(pads) != PADS_DIM:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                                 "pads", str(PADS_DIM), str(len(pads)))
    if len(dilations) != DILATION_DIM:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                                 "dilations", str(DILATION_DIM), str(len(dilations)))
    _check_dilations(dilations, dim_n, dim_c, dim_h, dim_w)
    _check_stride(strides, dim_n, dim_c)
    _check_shape(shape_in, shape_dedy, shape_dedw)
    strides = strides[dim_n], strides[dim_c], strides[dim_h], strides[dim_w]
    in_range, dedy_range = _check_range(in_range, dedy_range, shape_in, shape_dedy)

    filter_h_dilations = (shape_dedw[2] - 1) * dilations[2] + 1
    filter_w_dilations = (shape_dedw[3] - 1) * dilations[3] + 1
    fmap_h_min, fmap_w_min = FMAP_HW_MIN, FMAP_HW_MIN
    if DYNAMIC_FLAG in pads:
        pad_w = _align(shape_in[3], strides[3]) - strides[3] + \
            filter_w_dilations - shape_in[3]
        pad_w = tvm.max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = _align(shape_in[2], strides[2]) - strides[2] + \
            filter_h_dilations - shape_in[2]
        pad_h = tvm.max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        padding = [pad_up, pad_down, pad_left, pad_right]
    else:
        _check_attr_range_dw("pad's top", pads[0], PAD_MIN, PAD_MAX)
        _check_attr_range_dw("pad's bottom", pads[1], PAD_MIN, PAD_MAX)
        _check_attr_range_dw("pad's left", pads[2], PAD_MIN, PAD_MAX)
        _check_attr_range_dw("pad's right", pads[3], PAD_MIN, PAD_MAX)
        fmap_h_min = max(fmap_h_min, shape_dedw[2] - pads[0] - pads[1])
        fmap_w_min = max(fmap_w_min, shape_dedw[3] - pads[2] - pads[3])
        padding = pads
   
    if unknown_rank_flag or None in [*in_range[0], *in_range[2], *in_range[3]]:
        in_range[2] = (fmap_h_min, None)
        in_range[3] = (fmap_w_min, None)

    # filter value limit
    _check_attr_range_dw("filter's H", shape_dedw[2], FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range_dw("filter's W", shape_dedw[2], FILTER_HW_MIN, FILTER_HW_MAX)

    # Fmap value limit
    _check_attr_range_dw("Fmap's minH", in_range[2][0], fmap_h_min, FMAP_HW_MAX)
    _check_attr_range_dw("Fmap's minW", in_range[3][0], fmap_w_min, FMAP_HW_MAX)
    _check_attr_range_dw("Fmap's maxH", in_range[2][1], fmap_h_min, FMAP_HW_MAX)
    _check_attr_range_dw("Fmap's maxW", in_range[3][1], fmap_w_min, FMAP_HW_MAX)

    # stride value limit
    _check_attr_range_dw("stride's H", strides[2], STRIDE_HW_MIN, STRIDE_HW_MAX) 
    _check_attr_range_dw("stride's W", strides[3], STRIDE_HW_MIN, STRIDE_HW_MAX)

    fmap_n, fmap_c, fmap_h, fmap_w = shape_in
    _, _, dedy_h, dedy_w = shape_dedy
    fmap_c1 = (fmap_c + BLOCK_SIZE - 1) // BLOCK_SIZE
    fmap_shape_nc1hwc0 = (fmap_n, fmap_c1, fmap_h, fmap_w, BLOCK_SIZE)
    dedy_shape_nc1hwc0 = (fmap_n, fmap_c1, dedy_h, dedy_w, BLOCK_SIZE)
    fmap = tvm.placeholder(fmap_shape_nc1hwc0, name="fmap", dtype=in_dtype)
    filter_size = tvm.placeholder([4], name="filter_size", dtype="int32")
    dedy = tvm.placeholder(dedy_shape_nc1hwc0, name="dedy", dtype=dedy_dtype)

    org_tensors = {"input_fm": input_fm, "filter_size": filter_size, "out_backprop": out_backprop,
                   "filter_grad": filter_grad}

    para_dict = {
        "strides": [strides[DIM_H_NCHW], strides[DIM_W_NCHW]],
        "padding": padding,
        "dilations": dilations,
        "groups": fmap_c,
        "res_dtype": dedw_dtype,
        "kernel_name": kernel_name,
        "ori_tensors": org_tensors,
        "op_type": "DepthwiseConv2DBackpropFilter",
        "correct_range_flag": True
    }
    depthwise_conv2dbp_filter_para = CubeParaProcess(para_dict)
    depthwise_conv2dbp_filter_para.check_range_valid(shape_in, in_range, "", "NCHW")
    depthwise_conv2dbp_filter_para.check_range_valid(shape_dedy, dedy_range, "", "NHCW")
    dedy_range = _range_correction(in_range, shape_dedw, pads, strides, dilations, shape_dedy)
    #Dedy value limit
    _check_attr_range_dw("Dedy's minH inferenced from Fmap's minH",
                         dedy_range[2][0], DEDY_H_MIN, DEDY_HW_MAX)
    _check_attr_range_dw("Dedy's maxH inferenced from Fmap's maxH",
                         dedy_range[2][1], DEDY_H_MIN, DEDY_HW_MAX)
    _check_attr_range_dw("Dedy's minW inferenced from Fmap's minW",
                         dedy_range[3][0], DEDY_H_MIN, DEDY_HW_MAX)
    _check_attr_range_dw("Dedy's maxW inferenced from Fmap's maxW",
                         dedy_range[3][1], DEDY_H_MIN, DEDY_HW_MAX)

    dedw = tbe.conv2d_backprop_filter(
        input_x=fmap,
        out_backprop=dedy,
        filter_sizes =(shape_dedw[0]*shape_dedw[1], 1, shape_dedw[2], shape_dedw[3]),
        para_dict=para_dict
    )
    return {'op_placeholder': [fmap, filter_size, dedy], 'op_res':[dedw]}


def _get_pos_from_format(format_in):
    return {"pos_n": format_in.find("N"), "pos_c": format_in.find("C"), "pos_h": format_in.find("H"), \
            "pos_w": format_in.find("W")}


def _get_attrs(strides, pads, dilations, data_format):
    pos = _get_pos_from_format(data_format)
    pos_n = pos.get("pos_n")
    pos_c = pos.get("pos_c")
    pos_h = pos.get("pos_h")
    pos_w = pos.get("pos_w")
    dilations = [dilations[pos_n], dilations[pos_c],
                 dilations[pos_h], dilations[pos_w]]

    if len(strides) == 4:
        strides = [strides[pos_h], strides[pos_w]]

    return strides, pads, dilations


def _calc_max_fmap_w(input_fm, out_backprop, filter_grad, strides, dilations, pads, data_format):
    """
    modify max grade point, when exceed L1 size

    Parameters
    ----------
    same to conv2d_backprop_filter

    Returns
    -------
    modified dedx's w_range
    """
    input_ori_format = input_fm.get('ori_format')
    dedw_ori_format = filter_grad.get('ori_format')
    shape_in = input_fm.get('ori_shape')
    shape_dedy = out_backprop.get('ori_shape')
    shape_dedw = filter_grad.get('ori_shape')
    if dedw_ori_format in ('HWCK', 'HWCN'):
        dedw_nchw = (shape_dedw[3], shape_dedw[2], shape_dedw[0], shape_dedw[1])
    else:
        dedw_nchw = shape_dedw

    # index of origin dimension
    if input_ori_format == 'NHWC':
        dim_n, dim_h, dim_w, dim_c = 0, 1, 2, 3
        x_nchw = [shape_in[dim_n], shape_in[dim_c], shape_in[dim_h], shape_in[dim_w]]
        dedy_nchw = [shape_dedy[dim_n], shape_dedy[dim_c], shape_dedy[dim_h], shape_dedy[dim_w]]
    else:
        x_nchw = shape_in
        dedy_nchw = shape_dedy
    strides, pads, dilations = _get_attrs(strides, pads, dilations, data_format)
    pad_up, pad_down, pad_left, pad_right = pads
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations[DIM_H_NCHW], dilations[DIM_W_NCHW]
    filter_h_dilation = (dedw_nchw[DIM_H_NCHW] - 1) * dilation_h + 1
    filter_w_dilation = (dedw_nchw[DIM_W_NCHW] - 1) * dilation_w + 1
    al1_min_byte = C0_SIZE * C0_SIZE * 2
    if DYNAMIC_FLAG in pads:
        pad_h = _align(x_nchw[DIM_H_NCHW], stride_h) - stride_h + filter_h_dilation - x_nchw[DIM_H_NCHW]
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
    h_index = input_fm.get("ori_format").find("H")
    x_h_range = input_fm.get("ori_range")[h_index]      
    w_index = input_fm.get("ori_format").find("W")
    x_w_range = input_fm.get("ori_range")[w_index]
    upper_fmap_h_padding = x_h_range[1] + pad_up + pad_down
    lower_fmap_h_padding = x_h_range[0] + pad_up + pad_down
    is_conv1d_situation = upper_fmap_h_padding == 1 and lower_fmap_h_padding == 1 and \
                          filter_h_dilation == 1 and stride_h == 1
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")  # L1 size
    bl1_min_byte = l1_size - al1_min_byte
    if not is_conv1d_situation:
        kl1_min = bl1_min_byte // ((filter_h_dilation + stride_h) * C0_SIZE * 2)
        kl1_min_devided_sixteen = bl1_min_byte // (filter_h_dilation * C0_SIZE * 2)
        if DYNAMIC_FLAG in pads:
            max_dedy_devided_sixteen = _ceil(kl1_min_devided_sixteen, stride_w)
        else:
            max_dedy_devided_sixteen = (kl1_min_devided_sixteen + pad_left + \
                                        pad_right - filter_w_dilation) // stride_w + 1
        range_unchange = (dedy_nchw[DIM_H_NCHW] != 1 and x_w_range[1] <= kl1_min) or \
                         (dedy_nchw[DIM_H_NCHW] == 1 and x_w_range[1] <= kl1_min_devided_sixteen)
        if range_unchange:
            return x_h_range, x_w_range
        if dedy_nchw[DIM_H_NCHW] != 1 and x_nchw[DIM_W_NCHW] <= kl1_min:
            x_w_range = (x_w_range[0], min(kl1_min, x_w_range[1]))
        elif dedy_nchw[DIM_H_NCHW] == 1 and x_nchw[DIM_W_NCHW] <= kl1_min_devided_sixteen:
            x_h_range_low, x_h_range_high = _get_input(dedy_nchw[DIM_H_NCHW], 
                                                       filter_h_dilation, pads[:2], stride_h)
            x_h_range_low = max(x_h_range_low, x_h_range[0])
            x_h_range_high = min(x_h_range_high, x_h_range[1])
            x_h_range = (x_h_range_low, x_h_range_high)
            x_w_range = (x_w_range[0], min(kl1_min_devided_sixteen, x_w_range[1]))
        elif dedy_nchw[DIM_W_NCHW] % C0_SIZE == 0 and dedy_nchw[DIM_W_NCHW] <= max_dedy_devided_sixteen:
            x_w_range_low, x_w_range_high = _get_input(dedy_nchw[DIM_W_NCHW], 
                                                       filter_w_dilation, pads[2:], stride_w)            
            x_w_range_low = max(x_w_range_low, x_w_range[0])
            x_w_range_high = min(x_w_range_high, x_w_range[1])
            x_w_range = (x_w_range_low, x_w_range_high)
        else:
            error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                                       "Input is too large, the minimum tiling may exceed L1_Buffer")
    return x_h_range, x_w_range


@tbe_register.register_param_generalization("DepthwiseConv2DBackpropFilter")
def depthwise_conv2d_backprop_filter_generalization(input_fm,
                                                    filter_size,
                                                    out_backprop,
                                                    filter_grad,
                                                    strides,
                                                    dilations=(1, 1, 1, 1),
                                                    pads=(0, 0, 0, 0),
                                                    data_format='NHWC',
                                                    kernel_name="depthwise_conv2d_backprop_filter",
                                                    generalize_config=None):
    """
    depthwise_conv2d_backprop_filter generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to depthwise_conv2d_backprop_filter

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    support_mode = ["keep_rank"]
    if generalize_config["mode"] not in support_mode:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                                   "invalid generalize mode {}, only support {}".format(
                                                       str(generalize_config["mode"]), str(support_mode)))
    result = []
    if generalize_config["mode"] == "keep_rank":  # fuzz build situation
        # unknow_rank x ori_shape is [-2], others' shape length is 4
        unknow_rank = len(input_fm["ori_shape"]) == 1 and input_fm["ori_shape"][0] == -2
        if unknow_rank:
            error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                                       "not support unknow_rank under mode {}".format(
                                                           generalize_config["mode"]))
        have_range = {"input_fm": input_fm, "out_backprop": out_backprop}
        support_format = ["NCHW", "NHWC"]
        for name, tensor in have_range.items():
            # modify tesnors have range
            if tensor.get("ori_format") not in support_format:
                error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                                           "invalid {} ori_format {}, only support {}".format(
                                                               name, str(tensor.get("ori_format")),
                                                               str(support_format)))
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                                           "invalid {} ori_shape {}, only support {}d".format(
                                                               name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
            valid = isinstance(tensor.get("shape"), (list, tuple)) and len(tensor["shape"]) == SHAPE_LEN
            if not valid:
                error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
                                                           "invalid {} ori_shape {}, only support {}d".format(
                                                               name, str(tensor.get("shape")), str(SHAPE_LEN)))
            if name == "input_fm":
                x_h_range, x_w_range = _calc_max_fmap_w(input_fm, out_backprop, filter_grad, strides, dilations, pads, data_format)
                if x_w_range[0] < FMAP_HW_MAX:
                    tensor["ori_range"] = (tensor["ori_range"][0], tensor["ori_range"][1], x_h_range,
                                           x_w_range) if tensor.get("ori_format") == "NCHW" else (
                    tensor["ori_range"][0],
                    x_h_range, x_w_range, tensor["ori_range"][3])
                    # default format is NC1HWC0
                    tensor["range"] = (tensor["range"][0], tensor["range"][1], x_h_range,
                                       x_w_range, tensor["range"][4])
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
            tensor["shape"] = [-1, tensor["shape"][1], -1, -1, tensor["shape"][4]]
        result.append([input_fm, filter_size, out_backprop, filter_grad, strides, dilations, pads,
                       data_format, kernel_name])
    return result


@tbe_register.register_operator('DepthwiseConv2DBackpropFilter')
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (tuple, list), (tuple, list), str, str)
def depthwise_conv2d_backprop_filter(input_fm,
                                     filter_size,
                                     out_backprop,
                                     filter_grad,
                                     strides,
                                     dilations=(1, 1, 1, 1),
                                     pads=(0, 0, 0, 0),
                                     data_format='NHWC',
                                     kernel_name="depthwise_conv2d_backprop_filter"):
    """
    algorithm: depthwise_conv2d_backprop_filter

    calculating  depthwise convolution backward filter

    Parameters
    ----------
    input_fm : a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    filter_size : a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    out_backprop: a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    filter_grad : a dict.
        4-D origin shape of filter tensor [H, W, C, K],
        K is depthwise_multiplier, support float32.

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
    
    with tbe.compute():
        res = _depthwise_conv2dbp_filter_compute(
            input_fm, filter_size, out_backprop, filter_grad, strides, dilations,
            pads, data_format, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get('op_res'))
    # get real output tensor
    real_out = res.get('op_res')[0].op.input_tensors[0].op.input_tensors[0]   
    tensor_list = res.get('op_placeholder') + [real_out]
    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensor_list,
              'build_args': {'constant_realize_extent_in_infer_bound': False}}
    tbe.build(sch, config)
