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
dynamic conv2d_backprop_filter
"""
from __future__ import absolute_import

import warnings
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import tbe_register
from impl.util.util_cube_dynamic import correct_conv2d_backprop_range_start
from impl.util.util_cube_dynamic import gen_conv_shape_range


# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4
# the dim of strides in conv_backprop must be 2
STRIDES_SHAPE_DIM = 2
# the dim of pads in conv_backprop must be 4
PADDING_SHAPE_DIM = 4
# fmapH, fmapW must be in [1,4096]
FMAP_HW_MAX = 4096
FMAP_HW_MIN = 1

# DeDy H,W must be in [1,4096]
DEDY_HW_MAX = 4096
DEDY_W_MIN = 1
DEDY_H_MIN = 1

# filterH, filterW must be in [1,255]
FILTER_HW_MAX = 255
FILTER_HW_MIN = 1

# stride must be in [1,63]
STRIDE_HW_MAX = 63
STRIDE_HW_MIN = 1

# pad must be in [0,255]
PAD_MAX = 255
PAD_MIN = 0

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000
# the minimum dim of shape
DEFAULT_MIN_SHAPE_DIM = 1
# the max dim of shape
DEFAULT_MAX_SHAPE_DIM = 1

# the max size is 2**63-1
DATA_SIZE_MAX = 9223372036854775807

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}

# C0_SIZE
C0_SIZE = 16
# pads valid mode to be [0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0]
# If pads is string , only support "SAME" or "VALID"
PADDING_SUPPORT = ('SAME', 'VALID')

N_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3

DYNAMIC_FLAG = -1
DYNAMIC_RANK_FLAG = [-2]
RANGE_DIM_LEN = 2

ORI_SHAPE_LEN = 4
SHAPE_LEN = 5

L1FUSION_INPUT_CTR = 2
OP_TYPE = "conv2d_backprop_filter"


def get_op_support_info(x, filter_size, out_backprop, y, strides, pads,
                        dilations, groups=1, data_format='NHWC',
                        kernel_name="conv2d_backprop_filter"):
    """
    get the conv2d_backprop_filter split info

    """

    format_x = x.get("format")
    axis_reduce_list = None
    if format_x == "NC1HWC0":
        # only Cout1 can be cut without overlap
        axis_split_matrix = [
            [util_select_op_base.SplitInput([1, [1], [-1], [-1]]),
             util_select_op_base.SplitOutput([0, [1]])]
        ]
        axis_reduce_list = [
            [util_select_op_base.ReduceInput([0, [0]], [1, [0]]),
             util_select_op_base.ReduceOutput([0, 1, False])]
        ]
    else:
        axis_split_matrix = None
        axis_reduce_list = None

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, None)
    return op_cal_info_in_json


def _ceil(x_1, x_2):
    if x_2 == 0:
        error_manager_cube.raise_err_specific("conv2d_backprop_filter", "Division by zero")
    return (x_1 + x_2 - 1) // x_2


def _align(x_1, x_2):
    return _ceil(x_1, x_2) * x_2


def _get_pos_from_format(format_in):
    return {"pos_n": format_in.find("N"), "pos_c": format_in.find("C"), "pos_h": format_in.find("H"), \
        "pos_w": format_in.find("W")}


def _check_equal(x_1, x_2, param_1, param_2):
    if x_1 != x_2:
        error_manager_cube.raise_err_scene_equal_limitation("conv2d_backprop_filter", param_1,
                                                  param_2)


def _check_dimensions(shape, name, dimension):
    if len(shape) != dimension:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
            "{} should be {}d list".format(name, dimension))


def _check_type(shape, name, type_set):
    if not isinstance(shape, type_set):
        error_manager_cube.raise_err_specific("conv2d_backprop_filter",
            "type of {} should in {}".format(name, type_set))


def _check_data_format(data_format, name, format_set=None):
    if format_set is None:
        format_set=["NHWC", "NCHW"]
    if data_format not in format_set:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
            "format of {} should in {}".format(name, format_set))


def _get_nchw_shape(fmap, out_backprop, filters, groups):
    def _check_shape_rules():
        _check_type(dedy_shape, "out_backprop", (tuple, list))
        if list(dedy_shape) != DYNAMIC_RANK_FLAG:
            _check_dimensions(dedy_shape, "out_backprop", CONV_BACKPROP_SHAPE_DIM)

        _check_type(x_shape, "x", (tuple, list))
        if list(x_shape) != DYNAMIC_RANK_FLAG:
            _check_dimensions(x_shape, "x", CONV_BACKPROP_SHAPE_DIM)

        _check_type(dedw_shape, "y", (tuple, list))
        _check_dimensions(dedw_shape, "y", CONV_BACKPROP_SHAPE_DIM)

    def _get_shape(shape, ori_format):
        pos = _get_pos_from_format(ori_format)
        pos_n = pos.get("pos_n")
        pos_c = pos.get("pos_c")
        pos_h = pos.get("pos_h")
        pos_w = pos.get("pos_w")
        return [shape[pos_n], shape[pos_c], shape[pos_h], shape[pos_w]]

    x_shape = fmap.get("ori_shape")
    dedy_shape = out_backprop.get("ori_shape")
    dedw_shape = filters.get("ori_shape")
    x_format = fmap.get("ori_format")
    dedy_format = out_backprop.get("ori_format")
    dedw_format = filters.get("ori_format")
    x_range = fmap.get("range") if fmap.get("range") else fmap.get("ori_range")

    _check_shape_rules()

    dedw_shape = _get_shape(dedw_shape, dedw_format)
    cout, k_c, _, _ = dedw_shape
    cin = k_c * groups
    if list(dedy_shape) == DYNAMIC_RANK_FLAG:
        dedy_shape = [DYNAMIC_FLAG, cout, DYNAMIC_FLAG, DYNAMIC_FLAG]
    else:
        dedy_shape = _get_shape(dedy_shape, dedy_format)
        if dedy_shape[C_DIM] == DYNAMIC_FLAG:
            dedy_shape[C_DIM] = cout
    if list(x_shape) == DYNAMIC_RANK_FLAG:
        x_shape = [DYNAMIC_FLAG, cin, DYNAMIC_FLAG, DYNAMIC_FLAG]
        x_range = [(1, None), (cin, cin), (1, None), (1, None)]
    else:
        x_shape = _get_shape(x_shape, x_format)
        if x_shape[C_DIM] == DYNAMIC_FLAG:
            x_shape[C_DIM] = cin

        # get range
        if len(x_range) == 4:
            pos = _get_pos_from_format(x_format)
            pos_n = pos.get("pos_n")
            pos_c = pos.get("pos_c")
            pos_h = pos.get("pos_h")
            pos_w = pos.get("pos_w")
            x_range = [x_range[pos_n], x_range[pos_c], x_range[pos_h], x_range[pos_w]]
        elif len(x_range) == 5:
            x_range = [x_range[0], (x_shape[1], x_shape[1]), x_range[2], x_range[3]]
            x_range = [tuple(r) for r in x_range]
        else:
            error_manager_cube.raise_err_equal_invalid('conv2d_backprop_filter', 'range_format', 'in_format')

    ret = {"x_shape": x_shape, "dedy_shape": dedy_shape, "dedw_shape": dedw_shape, "x_range": x_range}
    return ret


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


def _calc_pads(fmap_shape, w_nchw, strides, dilations, pads):
    """
    calculate pads
    """
    # get pads
    stride_h, stride_w = strides
    _, _, dilation_h, dilation_w = dilations
    _, _, fmap_h, fmap_w, _ = fmap_shape
    _, _, filter_h, filter_w = w_nchw
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    if -1 in pads:
        pad_w = _align(fmap_w, stride_w) - stride_w + \
            filter_w_dilation - fmap_w
        pad_w = tvm.max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = _align(fmap_h, stride_h) - stride_h + \
            filter_h_dilation - fmap_h
        pad_h = tvm.max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pads = [pad_up, pad_down, pad_left, pad_right]
    else:
        pads = list(pads)
        pad_up, pad_down, pad_left, pad_right = pads
        if pad_up >= filter_h_dilation or pad_down >= filter_h_dilation:
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                "the height of pad can not be less than shape_filter's, \
                actual are {} and {}".format(pad_up, pad_down))

        if pad_left >= filter_w_dilation or pad_right >= filter_w_dilation:
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                "the width of pad can not be less than shape_filter's, \
                actual are {} and {}".format(pad_left, pad_right))
    return pads


def _check_const_dim(dim_value, dim_name):
    if not isinstance(dim_value, int):
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                 "the value of the {} dimension of shape must be int".format(dim_name))
    if dim_value <= 0:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                 "the value of the {} dimension of shape must be -1 or >0".format(dim_name))


def _get_input_shape(fmap_nchw, dedy_nchw, dedw_nchw):

    fmap_n, fmap_c, fmap_h, fmap_w = fmap_nchw
    dedy_n, dedy_c, dedy_h, dedy_w = dedy_nchw
    if fmap_n * dedy_n < 0:
        fmap_n = fmap_n if fmap_n > 0 else dedy_n
        dedy_n = fmap_n
    if fmap_n != dedy_n:
        error_manager_cube.raise_err_scene_equal_limitation("conv2d_backprop_filter", "Fmap's N",
                                                  "Dedy's N")
    if DYNAMIC_FLAG not in fmap_nchw:
        fmap_n = -1
    if DYNAMIC_FLAG in dedw_nchw:
        error_manager_cube.raise_err_specific_user(
            "conv2d_backprop_filter", "dynamic weight is not supported yet.")
    if fmap_nchw[C_DIM] == DYNAMIC_FLAG:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
            "dynamic c dimension is not supported yet.")

    dim_name_lis = ["N", "C", "H", "W"]
    for index, dim in enumerate(fmap_nchw):
        if dim != DYNAMIC_FLAG:
            _check_const_dim(dim, dim_name_lis[index])
    for index, dim in enumerate(dedy_nchw):
        if dim != DYNAMIC_FLAG:
            _check_const_dim(dim, dim_name_lis[index])

    fmap_shape = (fmap_n, fmap_c, fmap_h, fmap_w)
    dedy_shape = (fmap_n, dedy_c, dedy_h, dedy_w)
    return fmap_shape, dedy_shape


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


def _range_correction(fmap_range, kernel, pads, stride, dilation, out_shape):
    correct_range_flag = False
    fmap_range_n, fmap_range_c, fmap_range_h, fmap_range_w = fmap_range
    w_n, w_c, w_h, w_w = kernel
    if -1 in pads:
        # calculate dedy range for pad is SAME
        out_h_lower = _ceil(fmap_range_h[0], stride[0])
        out_h_upper = FMAP_HW_MAX
        if fmap_range_h[1]:
            out_h_upper = _ceil(fmap_range_h[1], stride[0])
        out_w_lower = _ceil(fmap_range_w[0], stride[1])
        out_w_upper = FMAP_HW_MAX
        if  fmap_range_w[1]:
            out_w_upper = _ceil(fmap_range_w[1], stride[1])
    else:
        # calcaulate output range for pad is list
        out_h_lower = _get_output(fmap_range_h[0], w_h,
                                  (pads[0], pads[1]), stride[0], dilation[2])
        if out_h_lower < 1:
            fmap_range_h_lower = min(max(w_h - pads[0] - pads[1], 1), fmap_range_h[1]) \
                                 if fmap_range_h[1] else max(w_h - pads[0] - pads[1], 1)
            fmap_range_h = (fmap_range_h_lower, fmap_range_h[1])
            out_h_lower = _get_output(fmap_range_h[0], w_h,
                                    (pads[0], pads[1]), stride[0], dilation[2])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input h " + \
                "range is less than 1, and the lower limit of the input h range is corrected " + \
                "as {}".format(fmap_range_h_lower))
        out_h_upper = FMAP_HW_MAX
        if fmap_range_h[1]:
            out_h_upper = _get_output(fmap_range_h[1], w_h,
                                    (pads[0], pads[1]), stride[0], dilation[2])

        out_w_lower = _get_output(fmap_range_w[0], w_w,
                                 (pads[2], pads[3]), stride[1], dilation[3])
        if out_w_lower < 1:
            fmap_range_w_lower = min(max(w_w - pads[2] - pads[3], 1), fmap_range_w[1]) \
                                 if fmap_range_w[1] else max(w_w - pads[2] - pads[3], 1)
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            out_w_lower = _get_output(fmap_range_w[0], w_w,
                                    (pads[2], pads[3]), stride[1], dilation[3])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input w " + \
                "range is less than 1, and the lower limit of the input w range is corrected " + \
                "as {}".format(fmap_range_w_lower))
        out_w_upper = FMAP_HW_MAX
        if fmap_range_w[1]:
            out_w_upper = _get_output(fmap_range_w[1], w_w,
                                 (pads[2], pads[3]), stride[1], dilation[3])
    dedy_range = [fmap_range_n, (out_shape[1], out_shape[1]),
                  (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]
    fmap_range = [fmap_range_n, fmap_range_c, fmap_range_h, fmap_range_w]

    return dedy_range, fmap_range, correct_range_flag


def _check_conv2dbp_filter_params(fmap_shape, dedy_shape, dedw_nchw, strides,
                                  pads, dilations, groups, fmap_dtype,
                                  dedy_dtype, dedw_dtype, kernel_name,
                                  fmap_range):

    def _check_attr_range_dw(name, value, attr_min=None, attr_max=None):
        if (not isinstance(value, int)) or value > attr_max \
                or value < attr_min:
            error_manager_cube.raise_err_attr_range_invalid(
                "conv2d_backprop_filter", [attr_min, attr_max], name, str(value))

    def _is_load3d_special():
        # limitation by chip:
        # load3d instruction not support out_w = 1
        # only Ascend310 and Hi3796CS can support
        if (
            tbe_platform.get_soc_spec("SOC_VERSION") in ["Ascend310", "Hi3796CV300CS", "SD3403"]
            and dedy_h != 1
            and dedy_w == 1
        ):
            return True
        return False

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype:
            bit_ratio = BIT_RATIO_DICT.get(dtype)
        else:
            bit_ratio = BIT_RATIO_DICT.get("float16")
        if attr_value * bit_ratio > DATA_SIZE_MAX:
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                "{} must be less than {}, but it is {}.".format(attr_name,
                DATA_SIZE_MAX, attr_value * bit_ratio))

    def _check_variable_range(range_i, mini, maxi=DATA_SIZE_MAX, name=None):
        """
        check variable range

        """
        if not isinstance(range_i, (tuple, list)):
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                "type of range must be tuple or list.")
        if len(range_i) != RANGE_DIM_LEN:
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                "each dimension of range must be 2.")
        if not isinstance(range_i[0], int):
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                "The lower limit of the range must be Int.")
        if range_i[1] and (not isinstance(range_i[1], int)):
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                "The upper limit of the range must be Int or None.")
        if range_i[0] < mini or range_i[0] > maxi:
            error_manager_cube.raise_err_attr_range_invalid(
                "conv2d_backprop_filter", [mini, maxi], name, range_i[0])
        if range_i[1] and (range_i[1] < mini or range_i[1] > maxi):
            error_manager_cube.raise_err_attr_range_invalid(
                "conv2d_backprop_filter", [mini, maxi], name, range_i[1])

    # First : Base check, Mainly required by interface appearance
    # ===========================================================
    # util check
    para_check.check_kernel_name(kernel_name)
    stride_h, stride_w = strides
    # stride value limit
    _check_attr_range_dw("stride's H", stride_h, STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range_dw("stride's W", stride_w, STRIDE_HW_MIN, STRIDE_HW_MAX)
    # check fmap_range
    batch_range, _, h_range, w_range = fmap_range
    batch_range = (max(batch_range[0], 1), batch_range[1])
    h_range = (max(h_range[0], 1), h_range[1])
    w_range = (max(w_range[0], 1), w_range[1])
    _check_variable_range(h_range, FMAP_HW_MIN, FMAP_HW_MAX, "fmap_h")
    _check_variable_range(w_range, FMAP_HW_MIN, FMAP_HW_MAX, "fmap_w")
    name_lis = ['fmap_batch', 'fmap_c']
    for index, dim_range in enumerate(fmap_range[:2]):
        _check_variable_range(dim_range, 1, name=name_lis[index])
    dedy_range, fmap_range, correct_range_flag = _range_correction(fmap_range,
        dedw_nchw, pads, strides, dilations, dedy_shape)

    lower_bound, upper_bound = zip(*fmap_range)
    # dtype check
    para_check.check_dtype_rule(fmap_dtype, ["float16"])
    para_check.check_dtype_rule(dedy_dtype, ["float16"])
    para_check.check_dtype_rule(dedw_dtype, ["float16", "float32"])

    # Second : Furture Check, Mainly required by SRS
    # ===========================================================
    # the relation limits between shape
    fmap_n, fmap_c, fmap_h, fmap_w = fmap_shape
    fmap_n, dedy_c, dedy_h, dedy_w = dedy_shape
    filter_n, filter_c, filter_h, filter_w = dedw_nchw
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    _, _, lower_fmap_h, lower_fmap_w = lower_bound
    upper_fmap_n, upper_fmap_c, upper_fmap_h, upper_fmap_w = upper_bound
    dedy_n_range, dedy_c_range, dedy_h_range, dedy_w_range = dedy_range
    dedy_n_range = (max(dedy_n_range[0], 1), dedy_n_range[1])
    dedy_c_range = (max(dedy_c_range[0], 1), dedy_c_range[1])
    dedy_h_range = (max(dedy_h_range[0], 1), dedy_h_range[1])
    dedy_w_range = (max(dedy_w_range[0], 1), dedy_w_range[1])


    _, pad, _ = _get_attrs(strides, pads, dilations, "NCHW")
    pad_up, pad_down, pad_left, pad_right = pad

    _, upper_pad, _ = _get_attrs(strides, pads, dilations, "NCHW")
    upper_pad_up, upper_pad_down, upper_pad_left, upper_pad_right = upper_pad
    upper_fmap_w_padding = None
    upper_fmap_h_padding = None
    if upper_fmap_w:
        upper_fmap_w_padding = upper_fmap_w + upper_pad_left + upper_pad_right
    if upper_fmap_h:
        upper_fmap_h_padding = upper_fmap_h + upper_pad_up + upper_pad_down

    _, lower_pad, _ = _get_attrs(strides, pads, dilations, "NCHW")
    lower_pad_up, lower_pad_down, lower_pad_left, lower_pad_right = lower_pad
    lower_fmap_w_padding = lower_fmap_w + lower_pad_left + lower_pad_right
    lower_fmap_h_padding = lower_fmap_h + lower_pad_up + lower_pad_down

    # special cases
    fmap_hw_max = FMAP_HW_MAX
    fmap_h_min, fmap_w_min = FMAP_HW_MIN, FMAP_HW_MIN
    dedy_hw_max = DEDY_HW_MAX
    dedy_hw_min = 2

    if _is_load3d_special():
        dedy_hw_min = 1

    if -1 not in pads:
        fmap_h_min = max(fmap_h_min, filter_h - pad[0] - pad[1])
        fmap_w_min = max(fmap_w_min, filter_w - pad[2] - pad[3])

    # filter value limit
    _check_attr_range_dw("filter's H", filter_h, FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range_dw("filter's W", filter_w, FILTER_HW_MIN, FILTER_HW_MAX)

    # Dedy value limit
    _check_variable_range(dedy_n_range, 1, name="dedy_n")
    _check_variable_range(dedy_c_range, 1, name="dedy_c")
    _check_variable_range(dedy_h_range, min(dedy_hw_min, DEDY_H_MIN), dedy_hw_max, "dedy_h")
    _check_variable_range(dedy_w_range, min(dedy_hw_min, DEDY_W_MIN), dedy_hw_max, "dedy_w")

    def _check_axis_hw(fmap_h, fmap_w, dedy_h, dedy_w):
        _check_equal(dedy_c, filter_n, "Dedy's C", "Filter's N")
        _check_equal(fmap_c, filter_c * groups, "Fmap's C", "Filter's C")
        if fmap_w * dedy_w < 0 and fmap_w == DYNAMIC_FLAG:
            dedy_w = DYNAMIC_FLAG
        calculated_dedy_w = _ceil(fmap_w, stride_w)
        if pad_left != -1 and pad_right != -1 and fmap_w != DYNAMIC_FLAG:
            if filter_w_dilation > upper_fmap_w_padding:
                error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                    "the w dim of Filter(after dilation) must be less than " + \
                    "the corresponding dim of input(after padding)")
            calculated_dedy_w = ((fmap_w - filter_w_dilation + int(pad_left) + int(pad_right)) //
                                stride_w) + 1
        if dedy_w == DYNAMIC_FLAG:
            dedy_w = calculated_dedy_w if fmap_w != DYNAMIC_FLAG else dedy_w
        elif calculated_dedy_w != dedy_w:
            error_manager_cube.raise_err_scene_equal_limitation("conv2d_backprop_filter",
                calculated_dedy_w, dedy_w)

        if fmap_h * dedy_h < 0 and fmap_h == DYNAMIC_FLAG:
            dedy_h = DYNAMIC_FLAG
        calculated_dedy_h = _ceil(fmap_h, stride_h)
        if pad_up != -1 and pad_down != -1 and fmap_h != DYNAMIC_FLAG:
            if filter_h_dilation > upper_fmap_h_padding:
                error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                    "the h dim of Filter(after dilation) must be less than " + \
                    "the corresponding dim of input(after padding)")
            calculated_dedy_h = ((fmap_h - filter_h_dilation + int(pad_up) + int(pad_down)) //
                                stride_h) + 1
        if dedy_h == DYNAMIC_FLAG:
            dedy_h = calculated_dedy_h if fmap_h != DYNAMIC_FLAG else dedy_h
        elif calculated_dedy_h != dedy_h:
            error_manager_cube.raise_err_scene_equal_limitation("conv2d_backprop_filter",
                calculated_dedy_h, dedy_h)
        return dedy_h, dedy_w

    dedy_h, dedy_w = _check_axis_hw(fmap_h, fmap_w, dedy_h, dedy_w)

    def _is_conv1d_situation():
        if upper_fmap_h_padding == 1 and lower_fmap_h_padding == 1 and filter_h_dilation == 1 and stride_h == 1:
            return True
        return False

    def _min_l1_byte():
        # Forth : L1 limitation, Mainly required by chip
        al1_min_byte = C0_SIZE * C0_SIZE * 2
        if _is_conv1d_situation():
            kl1_min = (C0_SIZE - 1) * stride_w + filter_w_dilation
        else:
            kl1_min = upper_fmap_w if upper_fmap_w else FMAP_HW_MAX
        if dedy_w_range[1] % C0_SIZE == 0:
            bl1_min_byte = filter_h_dilation * kl1_min * C0_SIZE * 2
        else:
            bl1_min_byte = (filter_h_dilation + stride_h) * kl1_min * C0_SIZE * 2
        l1_size = tbe_platform.get_soc_spec("L1_SIZE")  # L1 size
        if (al1_min_byte + bl1_min_byte) > l1_size:
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                "Input is too large, the minimum tiling may exceed L1_Buffer")

    _min_l1_byte()
    check_limitation_flag = True if (upper_fmap_n and upper_fmap_h and upper_fmap_w) else False
    if check_limitation_flag:
        upper_fmap_size = upper_fmap_n * _align(upper_fmap_c, C0_SIZE) * \
            upper_fmap_h * upper_fmap_w
        _check_64bits_limitation("fmap_size", upper_fmap_size, dtype=fmap_dtype)
    filter_size = _align(filter_n, C0_SIZE) * _align(filter_c, C0_SIZE) * \
        filter_h * filter_w
    if -1 not in pads and check_limitation_flag:
        upper_dedy_h = (upper_fmap_h + pad_up + pad_down - dilation_h *
                        (filter_h - 1) - 1) // stride_h + 1
        upper_dedy_w = (upper_fmap_w + pad_left + pad_right - dilation_w *
                        (filter_w - 1) - 1) // stride_w + 1
        upper_dedy_size = upper_fmap_n * _align(filter_n, C0_SIZE) * \
            upper_dedy_h * upper_dedy_w
        _check_64bits_limitation("dedy_size", upper_dedy_size,
                                 dtype=dedy_dtype)

    _check_64bits_limitation("filter_size", filter_size, dtype=dedw_dtype)

    fmap_shape = [fmap_n, _ceil(fmap_c, C0_SIZE), fmap_h, fmap_w, C0_SIZE]
    dedy_shape = [fmap_n, _ceil(dedy_c, C0_SIZE), dedy_h, dedy_w, C0_SIZE]
    results = {"fmap_shape": fmap_shape, "dedy_shape": dedy_shape, "dedy_range": dedy_range, 
               "fmap_range": fmap_range, "correct_range_flag": correct_range_flag}
    return results


def _check_and_config_para(x, out_backprop, y,
                           strides, pads, dilations,
                           groups, data_format, kernel_name):
    x_ori_shape = list(x.get("ori_shape"))
    dedy_ori_shape = list(out_backprop.get("ori_shape"))
    dedw_ori_shape = list(y.get("ori_shape"))
    x_dtype = x.get("dtype")
    dedy_dtype = out_backprop.get("dtype")
    dedw_dtype = y.get("dtype")
    x_format = x.get("ori_format")
    dedy_format = out_backprop.get("ori_format")
    dedw_format = y.get("ori_format")

    x_dtype = x_dtype.lower()
    dedy_dtype = dedy_dtype.lower()
    dedw_dtype = dedw_dtype.lower()
    _check_data_format(x_format, "x")
    _check_data_format(dedy_format, "out_backprop")
    _check_data_format(dedw_format, "res", ["NHWC", "NCHW", "HWCN"])
    _check_data_format(data_format, "data_format")

    if x_format != data_format:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter", "in_format != data_format")
    if dedy_format != data_format:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter", "in_format != data_format")
    if len(strides) != CONV_BACKPROP_SHAPE_DIM:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter", "strides should be 4d list")
    if len(dilations) != CONV_BACKPROP_SHAPE_DIM:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter", "dilations should be 4d list")

    if dilations[H_DIM] != 1 or dilations[W_DIM] != 1:
        error_manager_cube.raise_err_specific_user(
            "conv2d_backprop_filter", "dilations is not supported in dynamic shape yet.")

    if len(pads) != CONV_BACKPROP_SHAPE_DIM:
        error_manager_cube.raise_err_specific_user("conv2d_backprop_filter", "pads should be 4d list")

    para_check.check_shape_rule(dedw_ori_shape, min_dim=CONV_BACKPROP_SHAPE_DIM,
                                max_dim=CONV_BACKPROP_SHAPE_DIM)
    ret_nchw = _get_nchw_shape(x, out_backprop, y, groups)
    x_nchw = ret_nchw.get("x_shape")
    dedy_nchw = ret_nchw.get("dedy_shape")
    dedw_nchw = ret_nchw.get("dedw_shape")
    fmap_range = ret_nchw.get("x_range")

    fmap_shape, dedy_shape = \
        _get_input_shape(x_nchw, dedy_nchw, dedw_nchw)

    strides, pads, dilations = _get_attrs(strides, pads, dilations, data_format)
    results = _check_conv2dbp_filter_params(fmap_shape, dedy_shape, dedw_nchw, strides, pads, dilations,
                                            groups, x_dtype, dedy_dtype, dedw_dtype, kernel_name,
                                            fmap_range)
    fmap_shape = results.get("fmap_shape")
    dedy_shape = results.get("dedy_shape")
    dedy_range = results.get("dedy_range")
    fmap_range = results.get("fmap_range")
    correct_range_flag = results.get("correct_range_flag")

    config_dict = {
        "fmap_shape": fmap_shape,
        "dedy_shape": dedy_shape,
        "dedw_nchw": dedw_nchw,
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "fmap_range": fmap_range,
        "dedy_range": dedy_range,
        "correct_range_flag": correct_range_flag,
    }

    return config_dict


def _conv2d_backprop_filter_compute(x, filter_size, out_backprop, y,
                                    strides, pads, dilations,
                                    groups, data_format, kernel_name):
    x_dtype = x.get("dtype")
    dedy_dtype = out_backprop.get("dtype")
    dedw_dtype = y.get("dtype")
    x_dtype = x_dtype.lower()
    dedy_dtype = dedy_dtype.lower()
    dedw_dtype = dedw_dtype.lower()

    config_dict = _check_and_config_para(x, out_backprop, y,
                           strides, pads, dilations,
                           groups, data_format, kernel_name)

    fmap_shape = config_dict.get("fmap_shape")
    dedy_shape = config_dict.get("dedy_shape")
    dedw_nchw = config_dict.get("dedw_nchw")
    strides = config_dict.get("strides")
    pads = config_dict.get("pads")
    dilations = config_dict.get("dilations")
    fmap_range = config_dict.get("fmap_range")
    dedy_range = config_dict.get("dedy_range")
    correct_range_flag = config_dict.get("correct_range_flag")
    # define var
    if fmap_shape[N_DIM] == DYNAMIC_FLAG and dedy_shape[N_DIM] == DYNAMIC_FLAG:
        fmap_shape[N_DIM] = operation.var("batch", bound=fmap_range[0])
        dedy_shape[N_DIM] = fmap_shape[N_DIM]
        operation.add_exclude_bound_var(fmap_shape[N_DIM])
    if fmap_shape[H_DIM] == DYNAMIC_FLAG:
        fmap_shape[H_DIM] = operation.var("fmap_h", bound=fmap_range[2])
        operation.add_exclude_bound_var(fmap_shape[H_DIM])
        dedy_shape[H_DIM] = operation.var("dedy_h", bound=dedy_range[2])
        operation.add_exclude_bound_var(dedy_shape[H_DIM])
    if fmap_shape[W_DIM] == DYNAMIC_FLAG:
        fmap_shape[W_DIM] = operation.var("fmap_w", bound=fmap_range[3])
        operation.add_exclude_bound_var(fmap_shape[W_DIM])
        dedy_shape[W_DIM] = operation.var("dedy_w", bound=dedy_range[3])
        operation.add_exclude_bound_var(dedy_shape[W_DIM])

    fmap = tvm.placeholder(fmap_shape, name="fmap", dtype=x_dtype)
    filter_size = tvm.placeholder([4], name="filter_size", dtype="int32")
    dedy = tvm.placeholder(dedy_shape, name="dedy", dtype=dedy_dtype)

    pads = _calc_pads(fmap_shape, dedw_nchw, strides, dilations, pads)

    org_tensors = {"x": x, "filter_size": filter_size, "out_backprop": out_backprop, "y": y}

    para_dict = {
        "strides": strides,
        "padding": pads,
        "dilations": dilations,
        "groups": groups,
        "res_dtype": dedw_dtype,
        "kernel_name": kernel_name,
        "correct_range_flag": correct_range_flag,
        "ori_tensors": org_tensors,
    }

    dedw = tbe.conv2d_backprop_filter(input_x=fmap,
                                      out_backprop=dedy,
                                      filter_sizes=dedw_nchw,
                                      para_dict=para_dict)

    return {'op_placeholder': [fmap, filter_size, dedy], 'op_res': [dedw]}


def _calc_max_fmap_w(x, out_backprop, y, strides, dilations, pads, groups, data_format):
    """
    modify max grade point, when exceed L1 size

    Parameters
    ----------
    same to conv2d_backprop_filter

    Returns
    -------
    modified dedx's w_range
    """
    ret_nchw = _get_nchw_shape(x, out_backprop, y, groups)
    x_nchw = ret_nchw.get("x_shape")
    dedy_nchw = ret_nchw.get("dedy_shape")
    dedw_nchw = ret_nchw.get("dedw_shape")
    strides, pads, dilations = _get_attrs(strides, pads, dilations, data_format)
    pad_up, pad_down, pad_left, pad_right = pads
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations[H_DIM], dilations[W_DIM]
    filter_h_dilation = (dedw_nchw[H_DIM] - 1) * dilation_h + 1
    filter_w_dilation = (dedw_nchw[W_DIM] - 1) * dilation_w + 1
    w_index = x.get("ori_format").find("W")
    x_w_range = x.get("ori_range")[w_index]
    h_index = x.get("ori_format").find("H")
    x_h_range = x.get("ori_range")[h_index] 
    al1_min_byte = C0_SIZE * C0_SIZE * 2
    upper_fmap_h_padding = x_h_range[1] + pad_up + pad_down
    lower_fmap_h_padding = x_h_range[0] + pad_up + pad_down
    is_conv1d_situation = upper_fmap_h_padding == 1 and lower_fmap_h_padding == 1 and \
                          filter_h_dilation == 1 and stride_h == 1
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")  # L1 size
    bl1_min_byte = l1_size - al1_min_byte
    if not is_conv1d_situation:
        kl1_min = bl1_min_byte // ((filter_h_dilation + stride_h) * C0_SIZE * 2)
        kl1_min_devided_sixteen = bl1_min_byte // (filter_h_dilation * C0_SIZE * 2)
        max_dedy_devided_sixteen = (kl1_min_devided_sixteen + pad_left + \
                                    pad_right - filter_w_dilation) // stride_w + 1
        if kl1_min >= x_w_range[1]:
            x_w_range = x.get("ori_range")[w_index]
        elif x_nchw[W_DIM] <= kl1_min:
            x_w_range = (x_w_range[0], kl1_min)
        elif dedy_nchw[W_DIM] % C0_SIZE == 0 and dedy_nchw[W_DIM] <= max_dedy_devided_sixteen:
            x_w_range_low, x_w_range_high = _get_input(dedy_nchw[W_DIM], 
                                                       filter_w_dilation, pads[2:], stride_w)            
            x_w_range_low = max(x_w_range_low, x_w_range[0])
            x_w_range_high = min(x_w_range_high, x_w_range[1])
            x_w_range = (x_w_range_low, x_w_range_high)                 
        else:
            error_manager_cube.raise_err_specific_user("conv2d_backprop_filter",
                                                        "Input is too large, the minimum tiling may exceed L1_Buffer")
    return x_w_range


@tbe_register.register_param_generalization("Conv2DBackpropFilter")
def conv2d_bp_filter_generalization(x, filter_size, out_backprop, y, strides, pads, dilations,
                                    groups=1, data_format='NHWC', kernel_name="conv2d_backprop_filter",
                                    generalize_config=None):
    """
    conv2d_backprop_filter generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to conv2d_backprop_filter

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    support_mode = ["keep_rank"]
    if generalize_config["mode"] not in support_mode:
        error_manager_cube.raise_err_specific_user(OP_TYPE,
                                                   "invalid generalize mode {}, only support {}".format(
                                                       str(generalize_config["mode"]), str(support_mode)))
    result = []
    if generalize_config["mode"] == "keep_rank":  # fuzz build situation
        # unknow_rank x ori_shape is [-2], others' shape length is 4
        unknow_rank = len(x["ori_shape"]) == 1 and x["ori_shape"][0] == -2
        if unknow_rank:
            error_manager_cube.raise_err_specific_user(OP_TYPE,
                                                       "not support unknow_rank under mode {}".format(
                                                           generalize_config["mode"]))
        have_range = {"x": x, "out_backprop": out_backprop}
        support_format = ["NCHW", "NHWC"]
        for name, tensor in have_range.items():
            # modify tesnors have range
            if tensor.get("ori_format") not in support_format:
                error_manager_cube.raise_err_specific_user(OP_TYPE,
                                                           "invalid {} ori_format {}, only support {}".format(
                                                               name, str(tensor.get("ori_format")),
                                                               str(support_format)))
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                error_manager_cube.raise_err_specific_user(OP_TYPE,
                                                           "invalid {} ori_shape {}, only support {}d".format(
                                                               name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
            tensor = gen_conv_shape_range(tensor, OP_TYPE)
            if name == "x":
                tensor = correct_conv2d_backprop_range_start(tensor, y, dilations, pads, data_format)
                x_w_range = _calc_max_fmap_w(x, out_backprop, y, strides, dilations, pads, groups, data_format)
                if x_w_range[0] < FMAP_HW_MAX:
                    tensor["ori_range"] = (tensor["ori_range"][0], tensor["ori_range"][1], tensor["ori_range"][2], 
                                        x_w_range) if tensor.get("ori_format") == "NCHW" else (tensor["ori_range"][0],
                                        tensor["ori_range"][1], x_w_range, tensor["ori_range"][3])
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
        result.append([x, filter_size, out_backprop, y, strides, pads, dilations,
                       groups, data_format, kernel_name])
    return result


@register_operator('Conv2DBackpropFilter')
@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.KERNEL_NAME,
)
def conv2d_backprop_filter(x, filter_size, out_backprop, y, strides, pads,
                           dilations, groups=1,
                           data_format='NHWC',
                           kernel_name="conv2d_backprop_filter"):
    """
    algorithm: conv2d_backprop_filter

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        input feature map tensor.

    filter_size: tuple/list of 4 integers
        The shape of filter. 4-D with shape [filter_height, filter_width, in_channels,
        out_channels] or [out_channels, filter_height, filter_width, in_channels] or
        [out_channels, in_channel, filter_height, filter_width].

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        input weight tensor.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        output tensor, dtype must be assigned.

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    groups: int
            The number of filter's group. Default value is 1.

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    kernel_name: str
            kernel name, default value is "conv2d_backprop_filter"

    Returns
    -------
    None
    """

    with tbe.compute():
        res = _conv2d_backprop_filter_compute(
            x, filter_size, out_backprop, y, strides, pads, dilations,
            groups, data_format, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get('op_res'))

    # get real output tensor
    real_out = res.get('op_res')[0]
    tensor_list = res.get('op_placeholder') + [real_out]
    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensor_list,
              'build_args': {'constant_realize_extent_in_infer_bound': False}}
    tbe.build(sch, config)
