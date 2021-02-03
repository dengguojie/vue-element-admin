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

from te import tvm
import te.lang.cce as tbe
import te.platform as tbe_platform
import te.lang.base as tbe_base
from tbe.common.utils import para_check
from tbe.common.utils import errormgr


# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4
# the dim of strides in conv_backprop must be 2
STRIDES_SHAPE_DIM = 2
# the dim of pads in conv_backprop must be 4
PADDING_SHAPE_DIM = 4
# fmapH, fmapW must be in [1,4096]
FMAP_HW_MAX = 4096
FMAP_HW_MIN = 1

# DeDy H,W must be in [2,4096]
DEDY_HW_MAX = 4096
DEDY_HW_MIN = 2

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


def _ceil(x_1, x_2):
    if x_2 == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        raise RuntimeError(dict_args,
                           errormgr.get_error_message(dict_args))
    return (x_1 + x_2 - 1) // x_2


def _align(x_1, x_2):
    return _ceil(x_1, x_2) * x_2


def _get_pos_from_format(format_in):
    return format_in.find("N"), format_in.find("C"), format_in.find("H"), \
        format_in.find("W")


def _check_equal(x_1, x_2, param_1, param_2):
    if x_1 != x_2:
        dict_args = {}
        dict_args['errCode'] = "E64002"
        dict_args['param1'] = param_1
        dict_args['param2'] = param_2
        dict_args['actual_value'] = "{}, {}". \
                                    format(x_1, x_2)
        raise RuntimeError(dict_args,
                           errormgr.get_error_message(dict_args))


def _check_dimensions(shape, name, dimension):
    if len(shape) != dimension:
        dict_args = dict()
        dict_args["errCode"] = "E60107"
        dict_args["param_name"] = name
        raise RuntimeError(dict_args,
                           errormgr.get_error_message(dict_args))


def _check_type(shape, name, type_set):
    if not isinstance(shape, type_set):
        dict_args = dict()
        dict_args["errCode"] = "E60107"
        dict_args["param_name"] = name
        raise RuntimeError(dict_args,
                           errormgr.get_error_message(dict_args))


def _check_data_format(data_format, name, format_set=["NHWC", "NCHW"]):
    if data_format not in format_set:
        dict_args = {}
        dict_args['errCode'] = "E60008"
        dict_args['param_name'] = name
        dict_args['expected_format_list'] = format_set
        dict_args["format"] = data_format
        raise RuntimeError(dict_args,
                           errormgr.get_error_message(dict_args))


def _get_nchw_shape(fmap, out_backprop, filters):
    def _check_shape_rules():
        _check_type(dedy_shape, "out_backprop", (tuple, list))
        _check_dimensions(dedy_shape, "out_backprop", CONV_BACKPROP_SHAPE_DIM)

        _check_type(x_shape, "x", (tuple, list))
        _check_dimensions(x_shape, "x", CONV_BACKPROP_SHAPE_DIM)

        _check_type(dedw_shape, "y", (tuple, list))
        _check_dimensions(dedw_shape, "y", CONV_BACKPROP_SHAPE_DIM)

        _check_data_format(x_format, "x")
        _check_data_format(dedy_format, "out_backprop")
        _check_data_format(dedw_format, "res", ["NHWC", "NCHW", "HWCN"])

    def _get_shape(shape, ori_format):
        pos_n, pos_c, pos_h, pos_w = _get_pos_from_format(ori_format)
        return [shape[pos_n], shape[pos_c], shape[pos_h], shape[pos_w]]

    x_shape = fmap.get("ori_shape")
    dedy_shape = out_backprop.get("ori_shape")
    dedw_shape = filters.get("ori_shape")
    x_format = fmap.get("ori_format")
    dedy_format = out_backprop.get("ori_format")
    dedw_format = filters.get("ori_format")
    x_range = fmap.get("range")

    _check_shape_rules()

    x_shape = _get_shape(x_shape, x_format)
    dedy_shape = _get_shape(dedy_shape, dedy_format)
    dedw_shape = _get_shape(dedw_shape, dedw_format)

    # get range
    if len(x_range) == 4:
        pos_n, pos_c, pos_h, pos_w = _get_pos_from_format(x_format)
        x_range = [x_range[pos_n], x_range[pos_n], x_range[pos_h], x_range[pos_w]]
    elif len(x_range) == 5:
        x_range = [x_range[0], (x_shape[1], x_shape[1]), x_range[2], x_range[3]]
        x_range = [tuple(r) for r in x_range]
    else:
        raise RuntimeError("range format should be same as input format")
    for r in x_range:
        if r[0] > r[1]:
            raise RuntimeError("range lower bound should be less equal than \
                upper bound")

    return x_shape, dedy_shape, dedw_shape, x_range


def _get_attrs(strides, pads, dilations, data_format, fmap_shape, w_nchw):
    def _convert_shape_to_list(shape):
        for i, var in enumerate(shape):
            if isinstance(var, tvm.expr.IntImm):
                shape[i] = var.value

    def _check_attrs_rules():
        if len(strides) != 4 and len(strides) != 2:
            dict_args = dict()
            dict_args["errCode"] = "E60107"
            dict_args["param_name"] = "strides"
            raise RuntimeError(dict_args,
                               errormgr.get_error_message(dict_args))

        if isinstance(pads, (tuple, list)):
            _check_dimensions(pads, "pads", CONV_BACKPROP_SHAPE_DIM)

        _check_dimensions(dilations, "dilations", CONV_BACKPROP_SHAPE_DIM)
        _check_data_format(data_format, "data_format")

    _check_attrs_rules()

    pos_n, pos_c, pos_h, pos_w = _get_pos_from_format(data_format)
    dilations = [dilations[pos_n], dilations[pos_c],
                 dilations[pos_h], dilations[pos_w]]

    if len(strides) == 4:
        strides = [strides[pos_h], strides[pos_w]]

    # get pads
    stride_h, stride_w = strides
    _, _, dilation_h, dilation_w = dilations
    _, _, fmap_h, fmap_w = fmap_shape
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
        _convert_shape_to_list(pads)
    else:
        pads = list(pads)
        pad_up, pad_down, pad_left, pad_right = pads
        if pad_up >= filter_h_dilation or pad_down >= filter_h_dilation:
            dict_args = dict()
            dict_args["errCode"] = "E64005"
            dict_args["direction"] = 'H'
            dict_args["pads_dir"] = "pad_up and pad_down"
            dict_args["pads_value"] = "[{}, {}]".format(pad_up, pad_down)
            dict_args["filter_value"] = str(filter_h_dilation)
            raise RuntimeError(dict_args,
                               errormgr.get_error_message(dict_args))
        if pad_left >= filter_w_dilation or pad_right >= filter_w_dilation:
            dict_args = dict()
            dict_args["errCode"] = "E64005"
            dict_args["direction"] = 'W'
            dict_args["pads_dir"] = "pad_left and pad_right"
            dict_args["pads_value"] = "[{}, {}]".format(pad_left, pad_right)
            dict_args["filter_value"] = str(filter_w_dilation)
            raise RuntimeError(dict_args,
                               errormgr.get_error_message(dict_args))

    return strides, pads, dilations


def _get_input_shape(fmap_nchw, dedy_nchw, dedw_nchw, fmap_range):
    dynamic_mode = None

    fmap_n, fmap_c, fmap_h, fmap_w = fmap_nchw
    dedy_n, dedy_c, dedy_h, dedy_w = dedy_nchw
    if fmap_n != dedy_n:
        dict_args = {}
        dict_args['errCode'] = "E64002"
        dict_args['param1'] = "Fmap's N"
        dict_args['param2'] = "Dedy's N"
        dict_args['actual_value'] = "{}, {}".\
            format(fmap_n, dedy_n)
        raise RuntimeError(dict_args,
                           errormgr.get_error_message(dict_args))

    if fmap_nchw[2] == -1 and fmap_nchw[3] == -1 and -1 not in fmap_nchw[:2]:
        fmap_h = tbe_base.var("fmap_h", bound=fmap_range[2])
        fmap_w = tbe_base.var("fmap_w", bound=fmap_range[3])
        dedy_h = tbe_base.var("dedy_h")
        dedy_w = tbe_base.var("dedy_w")
        tbe_base.add_exclude_bound_var(fmap_h)
        tbe_base.add_exclude_bound_var(fmap_w)
        tbe_base.add_exclude_bound_var(dedy_h)
        tbe_base.add_exclude_bound_var(dedy_w)
        dynamic_mode = "dynamic_hw"
    elif fmap_nchw[0] == -1 and -1 not in fmap_nchw[1:]:
        fmap_n = tbe_base.var("batch", bound=fmap_range[0])
        tbe_base.add_exclude_bound_var(fmap_n)
        dynamic_mode = "dynamic_batch"
    else:
        dict_args = dict()
        dict_args["errCode"] = "E60108"
        dict_args["param_name"] = "out_backprop"
        dict_args["reason"] = "only support dynamic_hw and dynamic_batch now."
        raise RuntimeError(dict_args, errormgr.get_error_message(dict_args))

    fmap_shape = (fmap_n, fmap_c, fmap_h, fmap_w)
    dedy_shape = (fmap_n, dedy_c, dedy_h, dedy_w)
    return fmap_shape, dedy_shape, dynamic_mode


def _get_output(x_in, k_size, pads, stride, dilation):
    return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1


def _range_correction(fmap_range, kernel, pads, stride, dilation, out_shape):
    if -1 in pads:
        out_h_lower = _ceil(fmap_range[2][0], stride[0])
        out_h_upper = _ceil(fmap_range[2][1], stride[0])
        out_w_lower = _ceil(fmap_range[3][0], stride[1])
        out_w_upper = _ceil(fmap_range[3][1], stride[1])
    else:
        out_h_lower = _get_output(fmap_range[2][0], kernel[2],
                                  (pads[0], pads[1]), stride[0], dilation[2])
        out_h_upper = _get_output(fmap_range[2][1], kernel[2],
                                  (pads[0], pads[1]), stride[0], dilation[2])
        out_w_lower = _get_output(fmap_range[3][0], kernel[3],
                                  (pads[2], pads[3]), stride[1], dilation[3])
        out_w_upper = _get_output(fmap_range[3][1], kernel[3],
                                  (pads[2], pads[3]), stride[1], dilation[3])
    return [(out_shape[0], out_shape[0]), (out_shape[1], out_shape[1]),
            (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]


def _check_conv2dbp_filter_params(fmap_shape, dedy_shape, dedw_nchw, strides,
                                  pads, dilations, groups, fmap_dtype,
                                  dedy_dtype, dedw_dtype, kernel_name,
                                  dynamic_mode, fmap_range):

    def _check_attr_range_dw(name, value, attr_min=None, attr_max=None):
        if (not isinstance(value, int)) or value > attr_max \
                or value < attr_min:
            dict_args = {}
            dict_args["errCode"] = "E64001"
            dict_args["range"] = "[{},{}]".format(attr_min, attr_max)
            dict_args["attr_name"] = name
            dict_args["value"] = str(value)
            raise RuntimeError(dict_args,
                               errormgr.get_error_message(dict_args))

    def _is_load3d_special():
        # limitation by chip:
        # if kernel h,w in [1,11]
        # and fmap h/w after padding equals to filter h/w
        # load3d support h,w is 1
        if (1 <= filter_h <= 11) and (1 <= filter_w <= 11) \
            and (int(lower_fmap_h_padding) == filter_h or
                 int(lower_fmap_w_padding) == filter_w):
            return True
        return False

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype:
            bit_ratio = BIT_RATIO_DICT.get(dtype)
        else:
            bit_ratio = BIT_RATIO_DICT.get("float16")
        if attr_value * bit_ratio > DATA_SIZE_MAX:
            dict_args = {}
            dict_args['errCode'] = "E60020"
            dict_args['attr_name'] = attr_name
            raise RuntimeError(dict_args,
                               errormgr.get_error_message(dict_args))
    # First : Base check, Mainly required by interface appearance
    # ===========================================================
    # util check
    para_check.check_kernel_name(kernel_name)
    dedy_range = _range_correction(fmap_range, dedw_nchw, pads, strides,
                                   dilations, dedy_shape)
    lower_bound, upper_bound = zip(*fmap_range)
    lower_bound_dedy, upper_bound_dedy = zip(*dedy_range)
    para_check.check_shape_rule(lower_bound, CONV_BACKPROP_SHAPE_DIM,
                                CONV_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(upper_bound, CONV_BACKPROP_SHAPE_DIM,
                                CONV_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(dedw_nchw, CONV_BACKPROP_SHAPE_DIM,
                                CONV_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    # stride check
    para_check.check_shape_rule(strides,
                                STRIDES_SHAPE_DIM, STRIDES_SHAPE_DIM,
                                DEFAULT_MAX_SHAPE_NUM)
    # dilation check
    para_check.check_shape_rule(dilations,
                                CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                                DEFAULT_MAX_SHAPE_NUM)
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    _check_attr_range_dw("dilations's H", dilation_h,
                         DILATION_MIN, DILATION_MAX)
    _check_attr_range_dw("dilations's W", dilation_w,
                         DILATION_MIN, DILATION_MAX)
    # group check
    if groups != 1:
        dict_args = {
            'errCode': 'E50060',
            'op_name': 'dynamic conv2d_backprop_filter',
            'description': "only supports groups=1"
        }
        raise RuntimeError(dict_args,
                           errormgr.get_error_message(dict_args))

    if dilation_n != 1 or dilation_c != 1:
        dict_args = {}
        dict_args["errCode"] = "E60023"
        dict_args["dilation_n"] = str(dilation_n)
        dict_args["dilation_c"] = str(dilation_c)
        raise RuntimeError(dict_args,
                           errormgr.get_error_message(dict_args))

    # dtype check
    fmap_dtype = fmap_dtype.lower()
    dedy_dtype = dedy_dtype.lower()
    dedw_dtype = dedw_dtype.lower()

    para_check.check_dtype_rule(fmap_dtype, ["float16"])
    para_check.check_dtype_rule(dedy_dtype, ["float16"])
    para_check.check_dtype_rule(dedw_dtype, ["float16", "float32"])

    # Second : Furture Check, Mainly required by SRS
    # ===========================================================
    # the relation limits between shape
    fmap_n, fmap_c, fmap_h, fmap_w = fmap_shape
    fmap_n, dedy_c, dedy_h, dedy_w = dedy_shape
    filter_n, filter_c, filter_h, filter_w = dedw_nchw
    stride_h, stride_w = strides

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    _, _, lower_fmap_h, lower_fmap_w = lower_bound
    upper_fmap_n, upper_fmap_c, upper_fmap_h, upper_fmap_w = upper_bound
    _, _, lower_dedy_h, lower_dedy_w = lower_bound_dedy
    _, _, upper_dedy_h, upper_dedy_w = upper_bound_dedy

    _, pad, _ = _get_attrs(strides, pads, dilations,
                            "NCHW", fmap_shape, dedw_nchw)
    pad_up, pad_down, pad_left, pad_right = pad

    _, upper_pad, _ = _get_attrs(strides, pads, dilations, "NCHW",
                                 upper_bound, dedw_nchw)
    upper_pad_up, upper_pad_down, upper_pad_left, upper_pad_right = upper_pad
    upper_fmap_w_padding = upper_fmap_w + upper_pad_left + upper_pad_right
    upper_fmap_h_padding = upper_fmap_h + upper_pad_up + upper_pad_down

    _, lower_pad, _ = _get_attrs(strides, pads, dilations, "NCHW",
                                 lower_bound, dedw_nchw)
    lower_pad_up, lower_pad_down, lower_pad_left, lower_pad_right = lower_pad
    lower_fmap_w_padding = lower_fmap_w + lower_pad_left + lower_pad_right
    lower_fmap_h_padding = lower_fmap_h + lower_pad_up + lower_pad_down

    # special cases
    fmap_hw_max = FMAP_HW_MAX
    fmap_h_min, fmap_w_min = FMAP_HW_MIN, FMAP_HW_MIN
    dedy_hw_max = DEDY_HW_MAX
    dedy_hw_min = DEDY_HW_MIN

    if _is_load3d_special():
        dedy_hw_min = 1

    pads_status = -1 not in pads and sum(pads) != 0
    if dynamic_mode == "dynamic_hw" and pads_status:
        dict_args = {
            "errCode": "E60108",
            "attr_name": "pads",
            "reason": "pads is [-1,-1,-1,-1] or [0,0,0,0] when h or w dim is -1"
        }
        raise RuntimeError(
            dict_args, errormgr.get_error_message(dict_args)
        )

    if -1 not in pads:
        fmap_h_min = max(fmap_h_min, filter_h - pad[0] - pad[1])
        fmap_w_min = max(fmap_w_min, filter_w - pad[2] - pad[3])

    # filter value limit
    _check_attr_range_dw("filter's H", filter_h, FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range_dw("filter's W", filter_w, FILTER_HW_MIN, FILTER_HW_MAX)

    # Fmap value limit
    _check_attr_range_dw("Fmap's minH", lower_fmap_h, fmap_h_min, fmap_hw_max)
    _check_attr_range_dw("Fmap's minW", lower_fmap_w, fmap_w_min, fmap_hw_max)
    _check_attr_range_dw("Fmap's maxH", upper_fmap_h, fmap_h_min, fmap_hw_max)
    _check_attr_range_dw("Fmap's maxW", upper_fmap_w, fmap_w_min, fmap_hw_max)

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
    _check_attr_range_dw("stride's H", stride_h, STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range_dw("stride's W", stride_w, STRIDE_HW_MIN, STRIDE_HW_MAX)

    def _check_axis_hw():
        _check_equal(dedy_c, filter_n, "Dedy's C", "Filter's N")
        _check_equal(fmap_c, filter_c*groups, "Fmap's C", "Filter's C")
        if -1 not in pads:
            if filter_w_dilation > upper_fmap_w_padding:
                dict_args = dict()
                dict_args["errCode"] = "E60015"
                dict_args["w_of_x"] = str(upper_fmap_w_padding)
                dict_args["w_of_filter"] = str(filter_w_dilation)
                raise RuntimeError(dict_args,
                                   errormgr.get_error_message(dict_args))
            if filter_h_dilation > upper_fmap_h_padding:
                dict_args = dict()
                dict_args["errCode"] = "E60014"
                dict_args["min_h_of_x"] = str(upper_fmap_h_padding)
                dict_args["h_of_filter"] = str(filter_h_dilation)
                raise RuntimeError(dict_args,
                                   errormgr.get_error_message(dict_args))
        if dynamic_mode == "dynamic_batch":
            # Third : value check, Mainly required by the convolution rule
            if ((fmap_w - filter_w_dilation + int(pad_left) + int(pad_right)) //
                stride_w + 1) != dedy_w:
                dict_args = {}
                dict_args["errCode"] = "E60025"
                raise RuntimeError(dict_args,
                                   errormgr.get_error_message(dict_args))

            if ((fmap_h - filter_h_dilation + int(pad_up) + int(pad_down)) //
                    stride_h + 1) != dedy_h:
                dict_args = {}
                dict_args["errCode"] = "E60024"
                raise RuntimeError(dict_args,
                                   errormgr.get_error_message(dict_args))

    _check_axis_hw()

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
            kl1_min = upper_fmap_w
        if upper_dedy_w % C0_SIZE == 0:
            bl1_min_byte = filter_h_dilation * kl1_min * C0_SIZE * 2
        else:
            bl1_min_byte = (filter_h_dilation + stride_h) * kl1_min * C0_SIZE * 2
        l1_size = tbe_platform.get_soc_spec("L1_SIZE")  # L1 size
        if (al1_min_byte + bl1_min_byte) > l1_size:
            dict_args = {}
            dict_args["errCode"] = "E60108"
            dict_args["op_name"] = "conv2d_backprop_filter"
            dict_args["reason"] = \
                "for this input shape range, the minimum tiling may exceed \
                L1_Buffer, please lower the upper_bound of fmap_w and retry"
            raise RuntimeError(dict_args,
                               errormgr.get_error_message(dict_args))

    _min_l1_byte()

    upper_fmap_size = upper_fmap_n * _align(upper_fmap_c, C0_SIZE) * \
        upper_fmap_h * upper_fmap_w
    filter_size = _align(filter_n, C0_SIZE) * _align(filter_c, C0_SIZE) * \
        filter_h * filter_w
    _check_64bits_limitation("fmap_size", upper_fmap_size, dtype=fmap_dtype)
    if -1 not in pads:
        upper_dedy_h = (upper_fmap_h + pad_up + pad_down - dilation_h *
                        (filter_h - 1) - 1) // stride_h + 1
        upper_dedy_w = (upper_fmap_w + pad_left + pad_right - dilation_w *
                        (filter_w - 1) - 1) // stride_w + 1
        upper_dedy_size = upper_fmap_n * _align(filter_n, C0_SIZE) * \
            upper_dedy_h * upper_dedy_w
        _check_64bits_limitation("dedy_size", upper_dedy_size,
                                 dtype=dedy_dtype)

    _check_64bits_limitation("filter_size", filter_size, dtype=dedw_dtype)

    fmap_shape = (fmap_n, _ceil(fmap_c, C0_SIZE), fmap_h, fmap_w, C0_SIZE)
    dedy_shape = (fmap_n, _ceil(dedy_c, C0_SIZE), dedy_h, dedy_w, C0_SIZE)
    results = (fmap_shape, dedy_shape)
    return results


def _conv2d_backprop_filter_compute(x, filter_size, out_backprop, y,
                                    strides, pads, dilations,
                                    groups, data_format, kernel_name):
    x_dtype = x.get("dtype")
    dedy_dtype = out_backprop.get("dtype")
    dedw_dtype = y.get("dtype")

    x_nchw, dedy_nchw, dedw_nchw, fmap_range = _get_nchw_shape(x, out_backprop, y)
    fmap_shape, dedy_shape, dynamic_mode = \
        _get_input_shape(x_nchw, dedy_nchw, dedw_nchw, fmap_range)
    strides, pad, dilations = _get_attrs(strides, pads, dilations,
                                          data_format, fmap_shape, dedw_nchw)

    fmap_shape, dedy_shape = _check_conv2dbp_filter_params(
        fmap_shape, dedy_shape, dedw_nchw, strides, pads, dilations,
        groups, x_dtype, dedy_dtype, dedw_dtype, kernel_name, dynamic_mode,
        fmap_range)

    fmap = tvm.placeholder(fmap_shape, name="fmap", dtype=x_dtype)
    filter_size = tvm.placeholder([4], name="filter_size", dtype="int32")
    dedy = tvm.placeholder(dedy_shape, name="dedy", dtype=dedy_dtype)

    para_dict = {
        "strides": strides,
        "padding": pad,
        "dilations": dilations,
        "groups": groups,
        "res_dtype": dedw_dtype,
        "kernel_name": kernel_name
    }

    dedw = tbe.conv2d_backprop_filter_compute(
        input_x=fmap,
        out_backprop=dedy,
        filter_sizes=dedw_nchw,
        para_dict=para_dict
    )

    return {'op_placeholder': [fmap, filter_size, dedy], 'op_res': [dedw]}


@tbe_base.register_operator('Conv2DBackpropFilter')
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (tuple, list), (tuple, list), int, str, str)
def conv2d_backprop_filter(x, filter_size, out_backprop, y, strides, pads,
                           dilations=(1, 1, 1, 1), groups=1,
                           data_format='NHWC',
                           kernel_name="conv2d_backprop_filter"):
    """
    algorithm: conv2d_backprop_filter

    Parameters
    ----------
    x: dict with keys(shape, dtype and range)
       input feature map tensor

    filter_size: dict, will not be used

    out_backprop: dict with keys(shape and dtype)
                  out_backprop tensor

    y: dict with keys(shape and dtype)
       output tensor, dtype must be assigned

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_backprop_filter

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

    with tbe_base.compute():
        res = _conv2d_backprop_filter_compute(
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
