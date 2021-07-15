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
dynamic conv3d_backprop_filter
"""
from __future__ import absolute_import

import warnings

from impl.util import util_common
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


# the dim of shape in conv_backprop must be 5
_CONV_BACKPROP_SHAPE_DIM = 5
# the dim of strides in conv_backprop must be 3
_STRIDES_SHAPE_DIM = 3
# the dim of pads in conv_backprop must be 6
_PADDING_SHAPE_DIM = 6
# fmapH, fmapW must be in [1,4096]
_FMAP_HW_MAX = 4096
_FMAP_HW_MIN = 1

# DeDy H,W must be in [1,4096]
_DEDY_HW_MAX = 4096
_DEDY_HW_MIN = 1

# filterH, filterW must be in [1,255]
_FILTER_HW_MAX = 255
_FILTER_HW_MIN = 1

# stride must be in [1,63]
_STRIDE_HW_MAX = 63
_STRIDE_HW_MIN = 1

# pad must be in [0,255]
_PAD_MAX = 255
_PAD_MIN = 0

# dilation must be in [1,255]
_DILATION_MIN = 1
_DILATION_MAX = 255

# the max num of each axis of shape
_DEFAULT_MAX_SHAPE_NUM = 1000000
# the minimum dim of shape
_DEFAULT_MIN_SHAPE_DIM = 1
# the max dim of shape
_DEFAULT_MAX_SHAPE_DIM = 1

# the max size is 2**63-1
_DATA_SIZE_MAX = 9223372036854775807

# the bytes length of several dtype
_BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                   "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}

# C0_SIZE
_C0_SIZE = 16
# pads valid mode to be [0, 0, 0, 0]
_PADDING_VAILD = [0, 0, 0, 0, 0, 0]
# If pads is string , only support "SAME" or "VALID"
_PADDING_SUPPORT = ('SAME', 'VALID')
_DYNAMIC_RANK_FLAG = [-2]


def _get_pos_from_format(format_in):
    return (format_in.find("N"), format_in.find("D"), format_in.find("H"),
            format_in.find("W"), format_in.find("C"))


def _check_equal(x_1, x_2, param_1, param_2):
    if x_1 != x_2:
        dict_args = {}
        dict_args['errCode'] = "E64002"
        dict_args['param1'] = param_1
        dict_args['param2'] = param_2
        dict_args['actual_value'] = "{}, {}". \
                                    format(x_1, x_2)
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_dimensions(shape, name, dimension):
    if len(shape) != dimension:
        dict_args = {
            "errCode": "E60011",
            "attr_name": name,
            "range": "[{}, {}]".format(dimension, dimension),
            "value": "{}".format(len(shape))
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _get_ndhwc_shape(fmap, out_backprop, filters, dilations, strides, data_format, groups):
    def _check_shape_rules():
        if list(dedy_shape) != _DYNAMIC_RANK_FLAG:
            _check_dimensions(dedy_shape, "out_backprop", _CONV_BACKPROP_SHAPE_DIM)

        if list(x_shape) != _DYNAMIC_RANK_FLAG:
            _check_dimensions(x_shape, "x", _CONV_BACKPROP_SHAPE_DIM)

        para_check.check_shape_rule(strides, _CONV_BACKPROP_SHAPE_DIM, _CONV_BACKPROP_SHAPE_DIM)
        para_check.check_shape_rule(dilations, _CONV_BACKPROP_SHAPE_DIM, _CONV_BACKPROP_SHAPE_DIM)
        para_check.check_shape_rule(dedw_shape, _CONV_BACKPROP_SHAPE_DIM, _CONV_BACKPROP_SHAPE_DIM)

        para_check.check_format(x_format, ("NDHWC", "NCDHW"), "x")
        para_check.check_format(dedy_format, ("NDHWC", "NCDHW"), "out_backprop")
        para_check.check_format(dedw_format, ("NDHWC", "NCDHW", "DHWCN"), "y")
        para_check.check_format(data_format, ("NDHWC", "NCDHW"), "data_format")

        if x_format != data_format or dedy_format != data_format:
            error_manager_cube.raise_err_specific_user(
                "conv3d_backprop_filter",
                "The original format of fmap and out_backprop must be same as data_format.")

    def _get_shape(shape, ori_format):
        pos_n, pos_d, pos_h, pos_w, pos_c = _get_pos_from_format(ori_format)
        return [shape[pos_n], shape[pos_d], shape[pos_h], shape[pos_w], shape[pos_c]]

    x_shape = fmap.get("ori_shape")
    dedy_shape = out_backprop.get("ori_shape")
    dedw_shape = filters.get("ori_shape")
    x_format = fmap.get("ori_format")
    dedy_format = out_backprop.get("ori_format")
    dedw_format = filters.get("ori_format")
    x_range = fmap.get("range")

    _check_shape_rules()

    dedw_shape_ndhwc = _get_shape(dedw_shape, dedw_format)
    dilations_ndhwc = _get_shape(dilations, data_format)
    strides_ndhwc = _get_shape(strides, data_format)
    cout, _, _, _, dw_c = dedw_shape_ndhwc
    cin = dw_c * groups

    if list(dedy_shape) == _DYNAMIC_RANK_FLAG:
        dedy_shape_ndhwc = [-1, -1, -1, -1, cout]
    else:
        dedy_shape_ndhwc = _get_shape(dedy_shape, dedy_format)

    if list(x_shape) == _DYNAMIC_RANK_FLAG:
        x_shape_ndhwc = [-1, -1, -1, -1, cin]
        x_range = [(1, None), (1, None), (1, None), (1, None), (cin, cin)]
    else:
        x_shape_ndhwc = _get_shape(x_shape, x_format)
        # range dims maybe NDHWC, NDC1HWC0
        if len(x_range) == _CONV_BACKPROP_SHAPE_DIM:
            pos_n, pos_d, pos_h, pos_w, pos_c = _get_pos_from_format(x_format)
            x_range = [x_range[pos_n], x_range[pos_d], x_range[pos_h],
                       x_range[pos_w], x_range[pos_c]]
        elif len(x_range) == _CONV_BACKPROP_SHAPE_DIM + 1:
            x_range = [x_range[0], x_range[1], x_range[3], x_range[4], x_range[5]]
        else:
            raise RuntimeError("range format should be same as input format or ori_format")

        for i, r in enumerate(x_range):
            if x_shape_ndhwc[i] > 0:
                x_range[i] = (x_shape_ndhwc[i], x_shape_ndhwc[i])

            if r[1] and r[0] > r[1]:
                raise RuntimeError("range lower bound should be less equal than upper bound")

    x_shape_ndhwc[-1] = cin if x_shape_ndhwc[-1] == -1 else x_shape_ndhwc[-1]
    dedy_shape_ndhwc[-1] = cout if dedy_shape_ndhwc[-1] == -1 else dedy_shape_ndhwc[-1]

    ret = {"x_shape_ndhwc": x_shape_ndhwc,
           "dedy_shape_ndhwc": dedy_shape_ndhwc,
           "dedw_shape_ndhwc": dedw_shape_ndhwc,
           "x_range": x_range,
           "dilations_ndhwc": dilations_ndhwc,
           "strides_ndhwc": strides_ndhwc}

    return ret


def _check_pads_value(fmap_shape, pads):
    _check_dimensions(pads, "pads", _PADDING_SHAPE_DIM)

    if -1 not in fmap_shape[1:4]:
        for value in pads:
            if value < 0:
                error_manager_cube.raise_err_specific_user(
                    "conv3d_backprop_filter",
                    "Each value of pads has to be greater than 0 "
                    "when D/H/W demision is not dynamic.")
    else:
        for value in pads:
            if value < -1:
                error_manager_cube.raise_err_specific_user(
                    "conv3d_backprop_filter",
                    "Each value of pads must be -1 "
                    "when D/H/W demision is dynamic.")


def _get_pads_attr(strides, pads, dilations, fmap_shape, w_ndhwc):
    def _convert_shape_to_list(shape):
        for i, var in enumerate(shape):
            if isinstance(var, tvm.expr.IntImm):
                shape[i] = var.value

    # get pads
    _, stride_d, stride_h, stride_w, _ = strides
    _, dilation_d, dilation_h, dilation_w, _ = dilations
    _, fmap_d, fmap_h, fmap_w, _ = fmap_shape
    _, filter_d, filter_h, filter_w, _ = w_ndhwc
    filter_d_dilation = (filter_d - 1) * dilation_d + 1
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    pad_front, pad_back, pad_up, pad_down, pad_left, pad_right = pads
    if -1 in [pad_front, pad_back]:
        if fmap_d is not None:
            pad_d = util_common.align(fmap_d, stride_d) - stride_d + filter_d_dilation - fmap_d
            pad_d = tvm.max(pad_d, 0)
            pad_front = pad_d // 2
            pad_back = pad_d - pad_front
    else:
        if pad_front >= filter_d_dilation or pad_back >= filter_d_dilation:
            dict_args = dict()
            dict_args["errCode"] = "E64005"
            dict_args["direction"] = 'H'
            dict_args["pads_dir"] = "pad_front and pad_back"
            dict_args["pads_value"] = "[{}, {}]".format(pad_front, pad_back)
            dict_args["filter_value"] = str(filter_d_dilation)
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))
    if -1 in [pad_up, pad_down]:
        if fmap_h is not None:
            pad_h = util_common.align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
            pad_h = tvm.max(pad_h, 0)
            pad_up = pad_h // 2
            pad_down = pad_h - pad_up
    else:
        if pad_up >= filter_h_dilation or pad_down >= filter_h_dilation:
            dict_args = dict()
            dict_args["errCode"] = "E64005"
            dict_args["direction"] = 'H'
            dict_args["pads_dir"] = "pad_up and pad_down"
            dict_args["pads_value"] = "[{}, {}]".format(pad_up, pad_down)
            dict_args["filter_value"] = str(filter_h_dilation)
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))
    if -1 in [pad_left, pad_right]:
        if fmap_w is not None:
            pad_w = util_common.align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
            pad_w = tvm.max(pad_w, 0)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
    else:
        if pad_left >= filter_w_dilation or pad_right >= filter_w_dilation:
            dict_args = dict()
            dict_args["errCode"] = "E64005"
            dict_args["direction"] = 'W'
            dict_args["pads_dir"] = "pad_left and pad_right"
            dict_args["pads_value"] = "[{}, {}]".format(pad_left, pad_right)
            dict_args["filter_value"] = str(filter_w_dilation)
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    pads = [pad_front, pad_back, pad_up, pad_down, pad_left, pad_right]
    _convert_shape_to_list(pads)

    return pads


def _gen_input_shape(fmap_ndhwc, dedy_ndhwc, fmap_range):
    fmap_n, fmap_d, fmap_h, fmap_w, fmap_c = fmap_ndhwc
    dedy_n, dedy_d, dedy_h, dedy_w, dedy_c = dedy_ndhwc

    if fmap_n != dedy_n:
        if fmap_n != -1 and dedy_n == -1:
            dedy_n = fmap_n
        elif fmap_n == -1 and dedy_n != -1:
            fmap_n = dedy_n
        else:
            dict_args = {}
            dict_args['errCode'] = "E64002"
            dict_args['param1'] = "Fmap's N"
            dict_args['param2'] = "Dedy's N"
            dict_args['actual_value'] = "{}, {}".\
                format(fmap_n, dedy_n)
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    for value in fmap_ndhwc:
        if value < 1 and value != -1:
            error_manager_cube.raise_err_specific_user(
                "conv3d_backprop_filter",
                "Each dimension of fmap has to be -1 or positive integer.")

    if fmap_ndhwc[-1] == -1:
        error_manager_cube.raise_err_specific_user(
            "conv3d_backprop_filter", "dynamic c dimension is not supported yet.")

    if fmap_n == -1:
        fmap_n = operation.var("batch_n", bound=fmap_range[0])
        operation.add_exclude_bound_var(fmap_n)

    if fmap_d == -1:
        fmap_d = operation.var("fmap_d", bound=fmap_range[1])
        dedy_d = operation.var("dedy_d")
        operation.add_exclude_bound_var(fmap_d)
        operation.add_exclude_bound_var(dedy_d)

    if fmap_h == -1:
        fmap_h = operation.var("fmap_h", bound=fmap_range[2])
        dedy_h = operation.var("dedy_h")
        operation.add_exclude_bound_var(fmap_h)
        operation.add_exclude_bound_var(dedy_h)

    if fmap_w == -1:
        fmap_w = operation.var("fmap_w", bound=fmap_range[3])
        dedy_w = operation.var("dedy_w")
        operation.add_exclude_bound_var(fmap_w)
        operation.add_exclude_bound_var(dedy_w)

    fmap_shape = (fmap_n, fmap_d, fmap_h, fmap_w, fmap_c)
    dedy_shape = (fmap_n, dedy_d, dedy_h, dedy_w, dedy_c)
    return fmap_shape, dedy_shape


def _get_output(x_in, k_size, pads, stride, dilation):
    if not x_in:
        return None
    return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1


def _range_correction(fmap_range, kernel, pads, stride, dilation, out_shape):
    pad_front, pad_back, pad_up, pad_down, pad_left, pad_right = pads
    _, weight_d, weight_h, weight_w, _ = kernel
    _, dilation_d, dilation_h, dilation_w, _ = dilation
    filter_d_dilation = (weight_d - 1) * dilation_d + 1
    filter_h_dilation = (weight_h - 1) * dilation_h + 1
    filter_w_dilation = (weight_w - 1) * dilation_w + 1

    out_d_upper = None
    if -1 in [pad_front, pad_back]:
        out_d_lower = util_common.ceil(fmap_range[1][0], stride[1])
        if fmap_range[1][1]:
            out_d_upper = util_common.ceil(fmap_range[1][1], stride[1])
    else:
        out_d_lower = _get_output(fmap_range[1][0], weight_d,
                                  (pad_front, pad_back), stride[1], dilation[1])
        if out_d_lower < 1:
            fmap_range_d_lower = \
                min(filter_d_dilation, fmap_range[1][1]) if fmap_range[1][1] else filter_d_dilation
            fmap_range[1] = (fmap_range_d_lower, fmap_range[1][1])
            out_d_lower = _get_output(fmap_range[1][0], weight_d,
                                      (pad_front, pad_back), stride[1], dilation[1])
            warnings.warn("feature map range has been corrected due to invalid output shape D")
        if fmap_range[1][1]:
            out_d_upper = _get_output(fmap_range[1][1], weight_d,
                                      (pad_front, pad_back), stride[1], dilation[1])

    out_h_upper = None
    if -1 in [pad_up, pad_down]:
        out_h_lower = util_common.ceil(fmap_range[2][0], stride[2])
        if fmap_range[2][1]:
            out_h_upper = util_common.ceil(fmap_range[2][1], stride[2])
    else:
        out_h_lower = _get_output(fmap_range[2][0], weight_h,
                                  (pad_up, pad_down), stride[2], dilation[2])
        if out_h_lower < 1:
            fmap_range_h_lower = \
                min(filter_h_dilation, fmap_range[2][1]) if fmap_range[2][1] else filter_h_dilation
            fmap_range[2] = (fmap_range_h_lower, fmap_range[2][1])
            out_h_lower = _get_output(fmap_range[2][0], weight_h,
                                      (pad_up, pad_down), stride[2], dilation[2])
            warnings.warn("feature map range has been corrected due to invalid output shape H")
        if fmap_range[2][1]:
            out_h_upper = _get_output(fmap_range[2][1], weight_h,
                                      (pad_up, pad_down), stride[2], dilation[2])

    out_w_upper = None
    if -1 in [pad_left, pad_right]:
        out_w_lower = util_common.ceil(fmap_range[3][0], stride[3])
        if fmap_range[3][1]:
            out_w_upper = util_common.ceil(fmap_range[3][1], stride[3])
    else:
        out_w_lower = _get_output(fmap_range[3][0], weight_w,
                                  (pad_left, pad_right), stride[3], dilation[3])
        if out_w_lower < 1:
            fmap_range_w_lower = \
                min(filter_w_dilation, fmap_range[3][1]) if fmap_range[3][1] else filter_w_dilation
            fmap_range[3] = (fmap_range_w_lower, fmap_range[3][1])
            out_w_lower = _get_output(fmap_range[3][0], weight_w,
                                      (pad_left, pad_right), stride[3], dilation[3])
            warnings.warn("feature map range has been corrected due to invalid output shape W")
        if fmap_range[3][1]:
            out_w_upper = _get_output(fmap_range[3][1], weight_w,
                                      (pads[4], pads[5]), stride[3], dilation[3])

    dedy_range = [(fmap_range[0][0], fmap_range[0][1]),
                  (out_d_lower, out_d_upper), (out_h_lower, out_h_upper),
                  (out_w_lower, out_w_upper), (out_shape[-1], out_shape[-1])]

    return dedy_range, fmap_range


def _check_conv3dbp_filter_params(fmap_shape, dedy_shape, dedw_ndhwc, strides,
                                  pads, dilations, groups, fmap_dtype,
                                  dedy_dtype, dedw_dtype, kernel_name,
                                  fmap_range, dedy_range):
    def _check_attr_range_dw(name, value, attr_min=None, attr_max=None):
        if value is None:
            return
        if (not isinstance(value, int)) or value > attr_max \
                or value < attr_min:
            dict_args = {}
            dict_args["errCode"] = "E64001"
            dict_args["range"] = "[{},{}]".format(attr_min, attr_max)
            dict_args["attr_name"] = name
            dict_args["value"] = str(value)
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype:
            bit_ratio = _BIT_RATIO_DICT.get(dtype)
        else:
            bit_ratio = _BIT_RATIO_DICT.get("float16")
        if attr_value * bit_ratio > _DATA_SIZE_MAX:
            dict_args = {}
            dict_args['errCode'] = "E60020"
            dict_args['attr_name'] = attr_name
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    # First : Base check, Mainly required by interface appearance
    # ============================================================
    # util check
    para_check.check_kernel_name(kernel_name)
    lower_bound, upper_bound = zip(*fmap_range)
    lower_bound_dedy, upper_bound_dedy = zip(*dedy_range)
    para_check.check_shape_rule(lower_bound, _CONV_BACKPROP_SHAPE_DIM,
                                _CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    if None not in upper_bound:
        para_check.check_shape_rule(upper_bound, _CONV_BACKPROP_SHAPE_DIM,
                                    _CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)

    dilation_n, dilation_d, dilation_h, dilation_w, dilation_c = dilations
    _check_attr_range_dw("dilations's D", dilation_d,
                         _DILATION_MIN, _DILATION_MAX)
    _check_attr_range_dw("dilations's H", dilation_h,
                         _DILATION_MIN, _DILATION_MAX)
    _check_attr_range_dw("dilations's W", dilation_w,
                         _DILATION_MIN, _DILATION_MAX)

    if dilation_n != 1 or dilation_c != 1:
        dict_args = {}
        dict_args["errCode"] = "E60023"
        dict_args["dilation_n"] = str(dilation_n)
        dict_args["dilation_c"] = str(dilation_c)
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    # dtype check
    fmap_dtype = fmap_dtype.lower()
    dedy_dtype = dedy_dtype.lower()
    dedw_dtype = dedw_dtype.lower()

    para_check.check_dtype_rule(fmap_dtype, ("float16"))
    para_check.check_dtype_rule(dedy_dtype, ("float16"))
    para_check.check_dtype_rule(dedw_dtype, ("float16", "float32"))

    # Second : Future Check, Mainly required by SRS
    # =========================================================
    # the relation limits between shape
    fmap_n, fmap_d, fmap_h, fmap_w, fmap_c = fmap_shape
    _, dedy_d, dedy_h, dedy_w, dedy_c = dedy_shape
    filter_n, filter_d, filter_h, filter_w, filter_c = dedw_ndhwc
    _, stride_d, stride_h, stride_w, _ = strides

    filter_d_dilation = (filter_d - 1) * dilation_d + 1
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    _, lower_fmap_d, lower_fmap_h, lower_fmap_w, _ = lower_bound
    upper_fmap_n, upper_fmap_d, upper_fmap_h, upper_fmap_w, upper_fmap_c = upper_bound
    _, _, lower_dedy_h, lower_dedy_w, _ = lower_bound_dedy
    _, upper_dedy_d, upper_dedy_h, upper_dedy_w, _ = upper_bound_dedy

    # special cases
    if (upper_dedy_w and upper_dedy_h and lower_dedy_w <= 1 \
        and upper_dedy_w >= 1 and upper_dedy_h > 1):
        # Chip Design demand dedy_w must >=2 when dedy_h != 1
        dict_args = {
            'errCode': 'E62006',
            'error_desc': 'Chip Design demand dedy_w must >=2 when dedy_h != 1'
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    pad = _get_pads_attr(strides, pads, dilations,
                         fmap_shape, dedw_ndhwc)
    pad_front, pad_back, pad_up, pad_down, pad_left, pad_right = pad

    # special cases
    fmap_hw_max = _FMAP_HW_MAX
    fmap_h_min, fmap_w_min = _FMAP_HW_MIN, _FMAP_HW_MIN
    fmap_d_min = _FMAP_HW_MIN
    dedy_hw_max = _DEDY_HW_MAX
    dedy_hw_min = _DEDY_HW_MIN

    if -1 not in pads[:2]:
        fmap_d_min = max(fmap_d_min, filter_d_dilation - pad[0] - pad[1])
    if -1 not in pads[2:4]:
        fmap_h_min = max(fmap_h_min, filter_h_dilation - pad[2] - pad[3])
    if -1 not in pads[4:]:
        fmap_w_min = max(fmap_w_min, filter_w_dilation - pad[4] - pad[5])

    # filter value limit
    _check_attr_range_dw("filter's D", filter_d_dilation, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range_dw("filter's H", filter_h_dilation, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range_dw("filter's W", filter_w_dilation, _FILTER_HW_MIN, _FILTER_HW_MAX)

    # Fmap value limit
    _check_attr_range_dw("Fmap's minH", lower_fmap_h, fmap_h_min, fmap_hw_max)
    _check_attr_range_dw("Fmap's minW", lower_fmap_w, fmap_w_min, fmap_hw_max)
    _check_attr_range_dw("Fmap's minD", lower_fmap_d, fmap_d_min, fmap_hw_max)
    _check_attr_range_dw("Fmap's maxH", upper_fmap_h, fmap_h_min, fmap_hw_max)
    _check_attr_range_dw("Fmap's maxW", upper_fmap_w, fmap_w_min, fmap_hw_max)
    _check_attr_range_dw("Fmap's maxD", upper_fmap_d, fmap_d_min, fmap_hw_max)

    # Dedy value limit
    _check_attr_range_dw("Dedy's minH inferenced from Fmap's minH",
                         lower_dedy_h, dedy_hw_min, dedy_hw_max)
    _check_attr_range_dw("Dedy's minW inferenced from Fmap's minW",
                         lower_dedy_w, dedy_hw_min, dedy_hw_max)
    _check_attr_range_dw("Dedy's maxH inferenced from Fmap's maxH",
                         upper_dedy_h, dedy_hw_min, dedy_hw_max)
    _check_attr_range_dw("Dedy's maxW inferenced from Fmap's maxW",
                         upper_dedy_w, dedy_hw_min, dedy_hw_max)

    # stride value limit
    _check_attr_range_dw("stride's H", stride_h, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range_dw("stride's W", stride_w, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range_dw("stride's D", stride_d, _STRIDE_HW_MIN, _STRIDE_HW_MAX)

    def _check_axis_dhw():
        _check_equal(dedy_c, filter_n, "Dedy's C", "Filter's N")
        _check_equal(fmap_c, filter_c*groups, "Fmap's C", "Filter's C")
        if -1 not in pads[:2]:
            if (isinstance(fmap_d, int) and \
                ((fmap_d - filter_d_dilation + int(pad_front) + int(pad_back)) // stride_d + 1) != dedy_d):
                dict_args = {}
                dict_args["errCode"] = "E62508"
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))

        if -1 not in pads[2:4]:
            if (isinstance(fmap_h, int) and \
                ((fmap_h - filter_h_dilation + int(pad_up) + int(pad_down)) // stride_h + 1) != dedy_h):
                dict_args = {}
                dict_args["errCode"] = "E60024"
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
        if -1 not in pads[4:]:
            # Third : value check, Mainly required by the convolution rule
            if (isinstance(fmap_w, int) and \
                ((fmap_w - filter_w_dilation + int(pad_left) + int(pad_right)) // stride_w + 1) != dedy_w):
                dict_args = {}
                dict_args["errCode"] = "E60025"
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))

    _check_axis_dhw()

    def _min_l1_byte():
        if not upper_dedy_w or not upper_fmap_w:
            return
        # Forth : L1 limitation, Mainly required by chip
        al1_min_byte = _C0_SIZE * _C0_SIZE * 2
        if upper_dedy_w % _C0_SIZE == 0:
            bl1_min_byte = filter_h_dilation * upper_fmap_w * _C0_SIZE * 2
        else:
            bl1_min_byte = (filter_h_dilation + stride_h) * upper_fmap_w * _C0_SIZE * 2
        l1_size = tbe_platform.get_soc_spec("L1_SIZE") # L1 size
        if (al1_min_byte + bl1_min_byte) > l1_size:
            dict_args = {}
            dict_args["errCode"] = "E60108"
            dict_args["op_name"] = "conv2d_backprop_filter"
            dict_args["reason"] = \
                "for this input shape range, the minimum tiling may exceed \
                L1_Buffer, please lower the upper_bound of fmap_w and retry"
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    _min_l1_byte()

    if upper_fmap_n and upper_fmap_d and upper_fmap_h and upper_fmap_w:
        upper_fmap_size = (upper_fmap_n * util_common.align(upper_fmap_c, _C0_SIZE) *
                           upper_fmap_d * upper_fmap_h * upper_fmap_w)
        _check_64bits_limitation("fmap_size", upper_fmap_size, dtype=fmap_dtype)
        if -1 not in pads:
            upper_dedy_size = (upper_fmap_n * util_common.align(filter_n, _C0_SIZE) * \
                               upper_dedy_d * upper_dedy_h * upper_dedy_w)
            _check_64bits_limitation("dedy_size", upper_dedy_size,
                                     dtype=dedy_dtype)

    filter_size = (util_common.align(filter_n, _C0_SIZE) *
                   util_common.align(filter_c, _C0_SIZE) *
                   filter_d * filter_h * filter_w)
    _check_64bits_limitation("filter_size", filter_size, dtype=dedw_dtype)

    fmap_shape = (fmap_n, fmap_d, util_common.ceil(fmap_c, _C0_SIZE), fmap_h, fmap_w, _C0_SIZE)
    dedy_shape = (fmap_n, dedy_d, util_common.ceil(dedy_c, _C0_SIZE), dedy_h, dedy_w, _C0_SIZE)
    results = (fmap_shape, dedy_shape)
    return results


def _conv3d_backprop_filter_compute(x, filter_size, out_backprop, y,
                                    strides, pads, dilations,
                                    groups, data_format, kernel_name):
    x_dtype = x.get("dtype")
    dedy_dtype = out_backprop.get("dtype")
    dedw_dtype = y.get("dtype")

    shape_dict = _get_ndhwc_shape(x, out_backprop, y, dilations, strides, data_format, groups)

    x_ndhwc = shape_dict.get("x_shape_ndhwc")
    dedy_ndhwc = shape_dict.get("dedy_shape_ndhwc")
    dedw_ndhwc = shape_dict.get("dedw_shape_ndhwc")
    fmap_range = shape_dict.get("x_range")
    dilations = shape_dict.get("dilations_ndhwc")
    strides = shape_dict.get("strides_ndhwc")

    _check_pads_value(x_ndhwc, pads)

    # Do range Correction
    dedy_range, fmap_range = _range_correction(fmap_range, dedw_ndhwc, pads, strides,
                                               dilations, dedy_ndhwc)

    fmap_shape, dedy_shape = _gen_input_shape(x_ndhwc, dedy_ndhwc, fmap_range)

    group_dict = util_common.calculate_group(x_ndhwc[-1], dedy_ndhwc[-1],
                                             groups, _C0_SIZE, _C0_SIZE)

    pad = _get_pads_attr(strides, pads, dilations,
                         fmap_shape, dedw_ndhwc)

    fmap_shape, dedy_shape = _check_conv3dbp_filter_params(
        fmap_shape, dedy_shape, dedw_ndhwc, strides, pads, dilations,
        groups, x_dtype, dedy_dtype, dedw_dtype, kernel_name,
        fmap_range, dedy_range)

    fmap = tvm.placeholder(fmap_shape, name="fmap", dtype=x_dtype)
    filter_size = tvm.placeholder([_CONV_BACKPROP_SHAPE_DIM], name="filter_size", dtype="int32")
    dedy = tvm.placeholder(dedy_shape, name="dedy", dtype=dedy_dtype)
    strides_dhw = strides[1:4]
    dilations_ndchw = [dilations[0], dilations[1], dilations[4], dilations[2], dilations[3]]
    dedw_ndchw = [dedw_ndhwc[0], dedw_ndhwc[1], dedw_ndhwc[4], dedw_ndhwc[2], dedw_ndhwc[3]]
    para_dict = {
        "strides": strides_dhw,
        "pads": pad,
        "dilations": dilations_ndchw,
        "group_dict": group_dict,
        "res_dtype": dedw_dtype,
        "kernel_name": kernel_name
    }

    dedw = tbe.conv3d_backprop_filter(x=fmap, out_backprop=dedy, filter_size=dedw_ndchw, para_dict=para_dict)

    return {'op_placeholder': [fmap, filter_size, dedy], 'op_res': [dedw]}


@register_operator('Conv3DBackpropFilter')
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
def conv3d_backprop_filter(x, filter_size, out_backprop, y, strides, pads,
                           dilations=(1, 1, 1, 1, 1), groups=1,
                           data_format='NDHWC',
                           kernel_name="conv3d_backprop_filter"):
    """
    algorithm: conv3d_backprop_filter

    Parameters
    ----------
    x: dict with keys(shape, dtype and range)
       input feature map tensor

    filter_size: dict, will not be used

    out_backprop: dict with keys(shape and dtype)
                  out_backprop tensor

    y: dict with keys(shape and dtype)
       output tensor, dtype must be assigned

    strides: tuple/list of 5 integers
             filter move stride

    pads: tuple/list of 5 integers
          [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers
               filter expand size of dilated conv3d_backprop_filter

    groups: int
            The number of filter's group. Default value is 1.

    data_format: str
            An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "conv3d_backprop_filter"

    Returns
    -------
    None
    """

    with tbe.compute():
        res = _conv3d_backprop_filter_compute(
            x, filter_size, out_backprop, y, strides, pads, dilations,
            groups, data_format, kernel_name)

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
