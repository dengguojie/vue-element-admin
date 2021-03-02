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
conv3d_backprop_input
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
import te.lang.base as tbe_base
from tbe.dsl.compute import conv3d_backprop_input_compute as conv3d_bp_dx
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.errormgr import error_manager_cube as cube_err
from impl.util import util_common
from te import tvm


# the dim of shape in conv_backprop must be 5
_CONV_BACKPROP_SHAPE_DIM = 5
# the dim of pads in conv3d_backprop must be 6
_CONV_BACKPROP_PAD_SHAPE_DIM = 6
# the dim of strides in conv_backprop must be 5
_STRIDES_SHAPE_DIM = 5
# the dim of dilations in conv_backprop must be 5
_DILATIONS_SHAPE_DIM = 5

# fmapH, fmapW must be in [1,4096]
_FMAP_HW_MIN = 1
_FMAP_HW_MAX = 4096

# DeDy H,W must be in [1,4096]
_DEDY_HW_MIN = 1
_DEDY_HW_MAX = 4096

# filterH, filterW must be in [1,255]
_FILTER_HW_MIN = 1
_FILTER_HW_MAX = 255
_FILTER_HW_SIZE = 256
_FILTER_D_MAX = 128

# stride must be in [1,63] and h*w not lagger than 256
_STRIDE_HW_MIN = 1
_STRIDE_HW_MAX = 63
_STRIDE_SIZE_MAX = 256
_STRIDE_SIZE_HWD_MAX = 343

# special num
_KHWD_COEFF = 343

# the max num of each axis of shape
_DEFAULT_MAX_SHAPE_NUM = 1000000

# dilation must be in [1,255]
_DILATION_HW_MIN = 1
_DILATION_HW_MAX = 255

# NDHWC or NCDHW
FORMAT_5D_DIMS = 5
# NDC1HWC0
FORMAT_6D_DIMS = 6

# lower range
_LOWER_RANGE = 1
# upper range
_UPPER_RANGE = 4096

# the bytes length of several dtype
_BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                   "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
_DATA_SIZE_MAX = 9223372036854775807
_PADDING_VAILD = [0, 0, 0, 0, 0, 0]
# align with 16 for chips
_C0_SIZE = tbe_platform.C0_SIZE


def _check_attr_range(attr_name, attr_value, attr_min, attr_max):
    if attr_value < attr_min or attr_value > attr_max:
        cube_err.raise_err_attr_range_invalid(
            'conv3d_backprop_input', "[{},{}]".format(attr_min, attr_max),
            attr_name, str(attr_value))

def _get_ndhwc_shape(ori_format_filters, ori_shape_filters,
                     ori_format_out_backprop, ori_shape_out_backprop,
                     ori_shape_strides, ori_shape_dialtions, range_input,
                     ori_format_res, ori_shape_res):
    def _ncdhw2ndhwc(shape_ncdhw):
        shape_ndhwc = [shape_ncdhw[0], shape_ncdhw[2], shape_ncdhw[3], shape_ncdhw[4], shape_ncdhw[1]]
        return shape_ndhwc

    if ori_format_filters == "DHWCN":
        shape_filters = ori_shape_filters
    elif ori_format_filters == "NDHWC":
        shape_filters = (ori_shape_filters[1],
                         ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[0],
                         )
    elif ori_format_filters == "NCDHW":
        shape_filters = (ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[1],
                         ori_shape_filters[0],
                         )
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': '[{}, {}, {}]'
                .format('DHWCN', 'NDHWC', 'NCDHW'),
            'format': ori_format_filters
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if ori_format_out_backprop == "NDHWC":
        shape_out_backprop = list(ori_shape_out_backprop)
        shape_strides = ori_shape_strides
        shape_dilations = ori_shape_dialtions
    elif ori_format_out_backprop == "NCDHW":
        shape_out_backprop = _ncdhw2ndhwc(ori_shape_out_backprop)
        shape_strides = _ncdhw2ndhwc(ori_shape_strides)
        shape_dilations = _ncdhw2ndhwc(ori_shape_dialtions)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_out_backprop
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if ori_format_res == "NDHWC":
        shape_res = ori_shape_res
    elif ori_format_res == "NCDHW":
        shape_res = _ncdhw2ndhwc(ori_shape_res)
        if len(range_input) == FORMAT_5D_DIMS:
            range_input = _ncdhw2ndhwc(range_input)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_res
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    shape_out_backprop[-1] = shape_filters[-1]

    return shape_filters, shape_out_backprop, shape_strides, shape_dilations, range_input, shape_res

def _check_range(range, range_min=1, range_max=None):
    if range[0] < range_min:
        cube_err.raise_err_specific(
            'conv3d_backprop_input', "the lower bound of range should be larger than {}".format(range_min))
    if (range_max is not None) and (range[1] > range_max):
        cube_err.raise_err_specific(
            'conv3d_backprop_input', "the upper bound of range should be less than {}".format(range_max))
    if range[0] > range[1] and range[1] != -1:
        cube_err.raise_err_specific(
            'conv3d_backprop_input', "the upper bound of range should be larger than lower bound")

def _check_dynamic_flag(input_size_ndhwc):
    dim_str = "NDHW"
    for i in range(4):
        if input_size_ndhwc[i] < -1:
            cube_err.raise_err_specific(
                'conv3d_backprop_input',"Dynamic flag is -1, but dim {} is {}".format(dim_str[i], input_size_ndhwc[i]))
    if input_size_ndhwc[-1] < 0:
        cube_err.raise_err_specific(
            'conv3d_backprop_input',"Dim C does not support dynamic shape")

def _get_output(x_in, k_size, pads, stride, dilation):
    return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1

def _range_correction(fmap_range, kernel, pads, stride, dilation, out_shape):
    fmap_range_n, fmap_range_d, fmap_range_c1, fmap_range_h, fmap_range_w, fmap_range_c0 = fmap_range
    _check_range(fmap_range_n)
    _check_range(fmap_range_d)
    _check_range(fmap_range_h, _LOWER_RANGE, _UPPER_RANGE)
    _check_range(fmap_range_w, _LOWER_RANGE, _UPPER_RANGE)
    w_d, w_h, w_w, w_c, w_n = kernel
    _check_attr_range("stride's D", stride[1], _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's H", stride[2], _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's W", stride[3], _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    if not all(i == 0 for i in pads):
        out_d_lower = util_common.ceil(fmap_range_d[0], stride[1])
        out_d_upper = util_common.ceil(fmap_range_d[1], stride[1])
        out_h_lower = util_common.ceil(fmap_range_h[0], stride[2])
        out_h_upper = util_common.ceil(fmap_range_h[1], stride[2])
        out_w_lower = util_common.ceil(fmap_range_w[0], stride[3])
        out_w_upper = util_common.ceil(fmap_range_w[1], stride[3])
    else:
        out_d_lower = _get_output(fmap_range_d[0], w_d, (pads[0], pads[1]), stride[1], dilation[1])
        if out_d_lower < 1:
            fmap_range_d_lower = min(w_d, fmap_range_d[1]) if fmap_range_d[1] else w_d
            fmap_range_d = (fmap_range_d_lower, fmap_range_d[1])
            out_d_lower = _get_output(fmap_range_d[0], w_d, (pads[0], pads[1]), stride[1], dilation[1])
        out_d_upper = _get_output(fmap_range_d[1], w_d, (pads[0], pads[1]), stride[1], dilation[1])

        out_h_lower = _get_output(fmap_range_h[0], w_h, (pads[2], pads[3]), stride[2], dilation[2])
        if out_h_lower < 1:
            fmap_range_h_lower = min(w_h, fmap_range_h[1]) if fmap_range_h[1] else w_h
            fmap_range_h = (fmap_range_h_lower, fmap_range_h[1])
            out_h_lower = _get_output(fmap_range_h[0], w_h, (pads[2], pads[3]), stride[2], dilation[2])
        out_h_upper = _get_output(fmap_range_h[1], w_h, (pads[2], pads[3]), stride[2], dilation[2])

        out_w_lower = _get_output(fmap_range_w[0], w_w, (pads[4], pads[5]), stride[3], dilation[3])
        if out_w_lower < 1:
            fmap_range_w_lower = min(w_w, fmap_range_w[1]) if fmap_range_w[1] else w_w
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            out_w_lower = _get_output(fmap_range_w[0], w_w, (pads[4], pads[5]), stride[3], dilation[3])
        out_w_upper = _get_output(fmap_range_w[1], w_w, (pads[4], pads[5]), stride[3], dilation[3])

    range_dedy = [(fmap_range[0][0], fmap_range[0][1]), (out_d_lower, out_d_upper),
                  (util_common.ceil(out_shape[4], _C0_SIZE), util_common.ceil(out_shape[4], _C0_SIZE)),
                  (out_h_lower, out_h_upper), (out_w_lower, out_w_upper), (_C0_SIZE, _C0_SIZE)]

    range_input = [fmap_range_n, fmap_range_d, fmap_range_c1,
                   fmap_range_h, fmap_range_w, fmap_range_c0]

    return range_dedy, range_input

def _config_placeholder(shape_out_backprop, shape_filters, input_sizes, filters_dtype,
                        out_backprop_dtype, range_dedy, range_input):

    _, dy_k0, _ = tbe_platform.CUBE_MKN[out_backprop_dtype]['mac']
    _, w_k0, w_n0 = tbe_platform.CUBE_MKN[filters_dtype]['mac']

    dedy_batch, dedy_depth, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filters

    shape_filter_frac = (util_common.ceil(filter_channel, w_n0) * filter_depth * filter_h * filter_w,
                         util_common.ceil(filter_batch, w_k0), w_k0, w_n0)

    if input_sizes[0] == -1:
        dedy_batch = tbe_base.var("batch_n", range_input[0])
        tbe_base.add_exclude_bound_var(dedy_batch)
        input_sizes[0] = dedy_batch
    if input_sizes[1] == -1:
        dx_depth = tbe_base.var("dedx_d", range_input[1])
        dedy_depth = tbe_base.var("dedy_d", range_dedy[1])
        tbe_base.add_exclude_bound_var(dx_depth)
        tbe_base.add_exclude_bound_var(dedy_depth)
        input_sizes[1] = dx_depth
    if input_sizes[2] == -1:
        dx_h = tbe_base.var("dedx_h", range_input[3])
        dedy_h = tbe_base.var("dedy_h", range_dedy[3])
        tbe_base.add_exclude_bound_var(dx_h)
        tbe_base.add_exclude_bound_var(dedy_h)
        input_sizes[2] = dx_h
    if input_sizes[3] == -1:
        dx_w = tbe_base.var("dedx_w", range_input[4])
        dedy_w = tbe_base.var("dedy_w", range_dedy[4])
        tbe_base.add_exclude_bound_var(dx_w)
        tbe_base.add_exclude_bound_var(dedy_w)
        input_sizes[3] = dx_w

    shape_out_backprop = (dedy_batch, dedy_depth, dedy_h, dedy_w, dedy_channel)
    shape_dedy = (dedy_batch, dedy_depth, util_common.ceil(dedy_channel, dy_k0), dedy_h, dedy_w, dy_k0)

    dx_shape = tvm.placeholder([5], name="input_size", dtype="int32")
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    filter_frac = tvm.placeholder(shape_filter_frac, name="filter", dtype=filters_dtype)

    return dx_shape, dedy, filter_frac, input_sizes, shape_out_backprop

@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple), int,
                             (list, tuple), str, str, str, str,
                             (list, tuple), (list, tuple))
def check_conv3dbp_input_params(shape_filter,# pylint:disable=R0913,R0914,R0915
                                shape_out_backprop,
                                input_sizes, strides, pads, groups, dilations,
                                filter_dtype, out_backprop_dtype,
                                res_dtype, kernel_name, range_input, range_dedy):
    """
    The params check function of conv3d backprop input

    Parameters
    -------------------------
    shape_filter : The shape of filter
        5-D with shape (depth, height, weight, batch, channels)

    shape_out_backprop : The shape of gradients
        5-D with shape [batch, depth, height, weight,channels]

    input_sizes : The shape of feature map
        5-D with shape [batch, depth, height, weight, channels]

    strides : A list/tuple of ints. The stride of the sliding window

    pads : A list/tuple of ints or str

    groups : Int of blocked connections from input channels to output channels

    dilations : An optional list/tuple of ints

    filter_dtype : The dtype of filter data

    out_backprop_dtype : The dtype of gradients data

    res_dtype : The dtype of result(De/Dx) data

    kernel_name : Cce kernel name

    dynamic_mode : Dynamic mode

    Returns
    -----------------------
    All transformed params
    """
    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype is None:
            bit_ratio = _BIT_RATIO_DICT.get("float16")
        else:
            bit_ratio = _BIT_RATIO_DICT.get(dtype)
        if attr_value * bit_ratio > _DATA_SIZE_MAX:
            dict_args = {
                'errCode': 'E60020',
                'attr_name': attr_name,
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_ub_limitation():
        w_value = dedy_w_upper * stride_w

        aub_dedy_size_min = dedy_w_upper * _C0_SIZE * 2
        aub_filling_size_min = w_value * _C0_SIZE * 2
        cub_size_min = _C0_SIZE * _C0_SIZE * 2
        ub_size = tbe_platform.get_soc_spec("UB_SIZE")

        if (aub_dedy_size_min + aub_filling_size_min + cub_size_min) > ub_size:
            dict_args = {
                'errCode': 'E60119'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_l1_limitation():
        w_value = fmap_w_upper * stride_w
        if fmap_w_upper > _C0_SIZE:
            h_value_max = filter_h_dilation + 1
        elif _C0_SIZE % fmap_w_upper == 0:
            h_value_max = filter_h_dilation + _C0_SIZE // fmap_w_upper - 1
        else:
            h_value_max = filter_h_dilation + _C0_SIZE // fmap_w_upper + 1

        a_l1_size = h_value_max * w_value * ((filter_d_dilation - 2) // stride_d + 2) * _C0_SIZE * 2
        b_l1_size = filter_h_dilation * filter_w_dilation * filter_d_dilation * _C0_SIZE * _C0_SIZE * 2
        l1_size = tbe_platform.get_soc_spec("L1_SIZE")
        if (a_l1_size + b_l1_size) > l1_size:
            dict_args = {
                'errCode': 'E60026'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_shape_error():

        if not isinstance(fmap_batch, tvm.expr.Var) and dedy_channel != filter_batch:
            cube_err.raise_err_specific(
                'conv3d_backprop_input', "Shape error: Dedy's C must be equal to Filter'N.")

        # check dhw dimension
        if (not isinstance(fmap_h, tvm.expr.Var) and
            not isinstance(fmap_w, tvm.expr.Var) and
            not isinstance(fmap_deep, tvm.expr.Var)):
            pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right = pads
            fmap_h_padding = fmap_h + pad_up + pad_down
            fmap_w_padding = fmap_w + pad_left + pad_right
            fmap_d_padding = fmap_deep + pad_head + pad_tail
        
            if filter_h_dilation > fmap_h_padding:
                cube_err.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'H',
                                               str(filter_h_dilation),str(fmap_h_padding))
            if filter_w_dilation > fmap_w_padding:
                cube_err.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'W',
                                                str(filter_w_dilation), str(fmap_w_padding))
            if filter_d_dilation > fmap_d_padding:
                cube_err.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'D',
                                               str(filter_d_dilation), str(fmap_d_padding))
            if ((fmap_h - filter_h_dilation + pad_up + pad_down) // stride_h + 1) != dedy_h:
                dict_args = {'errCode': 'E60024',}
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
            if ((fmap_w - filter_w_dilation + pad_left + pad_right) // stride_w + 1) != dedy_w:
                dict_args = {'errCode': 'E60025',}
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
            if ((fmap_deep - filter_d_dilation + pad_head + pad_tail) // stride_d + 1) != dedy_deep:
                dict_args = {'errCode': 'E62508',}
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))

    # Base check, Mainly required by interface appearance
    # ===========================================================
    # para_check check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_filter, _CONV_BACKPROP_SHAPE_DIM,
                                _CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(strides, _STRIDES_SHAPE_DIM, _STRIDES_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(dilations, _DILATIONS_SHAPE_DIM,
                                _DILATIONS_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)

    # pads check
    if isinstance(pads, (tuple, list)) and len(pads) != _CONV_BACKPROP_PAD_SHAPE_DIM:
        cube_err.raise_err_one_para('E62501', 'conv3d_backprop_input', 'pads')

    if isinstance(pads, str) and pads not in ['SAME', 'VALID']:
        cube_err.raise_err_input_params_not_expected(
            'conv3d_backprop_input', 'pads', 'SAME or VALID', str(pads))

    #dilations check
    dilation_n, dilation_d, dilation_h, dilation_w, dilation_c = dilations
    if dilation_d != 1:
        cube_err.raise_err_specific('conv3d_backprop_input', "dilation in D dimension only supports 1.")

    # dtype check
    filter_dtype = filter_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    para_check.check_dtype_rule(filter_dtype, ['float16'], "filter")
    para_check.check_dtype_rule(out_backprop_dtype, ['float16'], "out_backprop")
    para_check.check_dtype_rule(res_dtype, ['float16'], "output")

    # the relation limits between shape
    shape_filter = list(shape_filter)
    shape_out_backprop = list(shape_out_backprop)
    input_sizes = list(input_sizes)
    strides = list(strides)
    fmap_batch, fmap_deep, fmap_h, fmap_w, fmap_channel = input_sizes
    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    _, stride_d, stride_h, stride_w, _ = strides

    group_dict = util_common.calculate_group(fmap_channel, filter_batch, groups, _C0_SIZE, _C0_SIZE)

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    filter_d_dilation = (filter_depth - 1) * dilation_d + 1

    pad_var_flag = False
    if not isinstance(pads, str):
        pad_var_flag = all(i == -1 for i in pads)
        pad_all_positive_flag = all(i >= 0 for i in pads)
        if not pad_var_flag and not pad_all_positive_flag:
            cube_err.raise_err_specific(
                    'conv3d_backprop_input', "pad should be positive")
    # pads compute
    if pads == 'SAME' or pad_var_flag:
        pad_h = util_common.align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = tvm.max(pad_h, 0) if isinstance(fmap_h, tvm.expr.Var) else max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = util_common.align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = tvm.max(pad_w, 0) if isinstance(fmap_w, tvm.expr.Var) else max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_d = util_common.align(fmap_deep, stride_d) - stride_d + filter_d_dilation - fmap_deep
        pad_d = tvm.max(pad_d, 0) if isinstance(fmap_deep, tvm.expr.Var) else max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head
        pads = [pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = _PADDING_VAILD

    pads = list(pads)
    if isinstance(fmap_deep, tvm.expr.Var):
        fmap_d_upper, dedy_d_upper = range_input[1][1], range_dedy[1][1]
    else:
        fmap_d_upper, dedy_d_upper = fmap_deep, dedy_deep
    if isinstance(fmap_h, tvm.expr.Var):
        fmap_h_upper, dedy_h_upper = range_input[3][1], range_dedy[3][1]
        fmap_h_lower, dedy_h_lower = range_input[3][0], range_dedy[3][0]
    else:
        fmap_h_upper, dedy_h_upper = fmap_h, dedy_h
        fmap_h_lower, dedy_h_lower = fmap_h, dedy_h
    if isinstance(fmap_w, tvm.expr.Var):
        fmap_w_upper, dedy_w_upper = range_input[4][1], range_dedy[4][1]
        fmap_w_lower, dedy_w_lower = range_input[4][0], range_dedy[4][0]
    else:
        fmap_w_upper, dedy_w_upper = fmap_w, dedy_w
        fmap_w_lower, dedy_w_lower = fmap_w, dedy_w
    if isinstance(fmap_batch, tvm.expr.Var):
        fmap_batch_upper, dedy_batch_upper = range_input[0][1], range_dedy[0][1]
    else:
        fmap_batch_upper, dedy_batch_upper = fmap_batch, dedy_batch

    if fmap_h_upper != 1 and fmap_w_upper == 1:
        # Chip Design demand fmap_w must larger than 2 when fmap_h == 1
        cube_err.raise_err_one_para(
            'E62006', 'conv3d_backprop_input', 'Chip Design demand input_size_w must >=2 when input_size_h != 1')

    _check_shape_error()

    _check_l1_limitation()

    if stride_h > 1 or stride_w > 1:
        _check_ub_limitation()

    # Dedy value limit
    _check_attr_range("Dedy's H after expands", dedy_h_lower * stride_h,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)
    _check_attr_range("Dedy's H after expands", dedy_h_upper * stride_h,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)
    _check_attr_range("Dedy's W after expands", dedy_w_lower * stride_w,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)
    _check_attr_range("Dedy's W after expands", dedy_w_upper * stride_w,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)

    # filter value limit
    _check_attr_range("filter's H", filter_h, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range("filter's W", filter_w, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range("filter's D", filter_depth, _FILTER_HW_MIN, _FILTER_D_MAX)
    _check_attr_range("filter H*W", filter_h * filter_w, _FILTER_HW_MIN,
                      _FILTER_HW_SIZE)
    _check_attr_range("filter H*W*D", filter_h * filter_w * filter_depth,
                      _FILTER_HW_MIN, _KHWD_COEFF)

    # Fmap value limit
    _check_attr_range("Fmap's H", fmap_h_lower, _FMAP_HW_MIN, _FMAP_HW_MAX)
    _check_attr_range("Fmap's W", fmap_w_lower, _FMAP_HW_MIN, _FMAP_HW_MAX)
    _check_attr_range("Fmap's H", fmap_h_upper, _FMAP_HW_MIN, _FMAP_HW_MAX)
    _check_attr_range("Fmap's W", fmap_h_upper, _FMAP_HW_MIN, _FMAP_HW_MAX)
    # stride value limit
    _check_attr_range("stride's H*W",
                      stride_h * stride_w, _STRIDE_HW_MIN, _STRIDE_SIZE_MAX)
    _check_attr_range("stride's H*W*D", stride_h * stride_w * stride_d,
                      _STRIDE_HW_MIN, _STRIDE_SIZE_HWD_MAX)

    # dilation value limit
    _check_attr_range("dilation's N", dilation_n, _DILATION_HW_MIN, _DILATION_HW_MIN)
    _check_attr_range("dilation's C", dilation_c, _DILATION_HW_MIN, _DILATION_HW_MIN)
    _check_attr_range("dilation's H", dilation_h, _DILATION_HW_MIN, _DILATION_HW_MAX)
    _check_attr_range("dilation's W", dilation_w, _DILATION_HW_MIN, _DILATION_HW_MAX)

    # check shape size, 64 bits limitation
    # ===========================================================
    fmap_size = (fmap_batch_upper * util_common.align(fmap_channel, _C0_SIZE) *
                 fmap_d_upper * fmap_h_upper * fmap_w_upper)
    dedy_size = (dedy_batch_upper * dedy_channel * dedy_d_upper *
                 dedy_h_upper * dedy_w_upper)
    filter_size = util_common.align(filter_batch, _C0_SIZE) * util_common.align(
                  filter_channel, _C0_SIZE) * filter_depth * filter_h * filter_w
    _check_64bits_limitation("input", fmap_size, dtype=res_dtype)
    _check_64bits_limitation("out_backprop", dedy_size, dtype=out_backprop_dtype)
    _check_64bits_limitation("filter", filter_size, dtype=filter_dtype)

    result = (shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
              filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict)

    return result

def check_and_config_para(filter, out_backprop, y, input_size, strides, pads,
                           dilations, groups, data_format, kernel_name):

    ori_shape_filters = filter.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    # range_input is N,D,C1,H,W,C0
    range_input = y.get("range")
    filters_dtype = filter.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y.get("dtype")
    ori_format_filters = filter.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")
    ori_format_res = y.get("ori_format")

    if not(ori_format_res == ori_format_out_backprop == data_format):
        cube_err.raise_err_specific(
            'conv3d_backprop_input',"The data format of out_backprop, input_size and data_format should be same")

    ori_shape_strides = strides
    ori_shape_dilations = dilations

    input_sizes = list(y.get("ori_shape"))

    shape_filters, shape_out_backprop, shape_strides, shape_dilations, range_input, input_sizes = \
        _get_ndhwc_shape(ori_format_filters,
                         ori_shape_filters,
                         ori_format_out_backprop,
                         ori_shape_out_backprop,
                         ori_shape_strides,
                         ori_shape_dilations,
                         range_input,
                         ori_format_res,
                         input_sizes)
    _check_dynamic_flag(input_sizes)
    if len(range_input) == FORMAT_5D_DIMS:
        c1_value = util_common.ceil(input_sizes[-1], _C0_SIZE)
        range_input = [range_input[0], range_input[1], (c1_value, c1_value),
                       range_input[2], range_input[3], (_C0_SIZE, _C0_SIZE)]
    # get range_dedy
    range_dedy, range_input = _range_correction(range_input, shape_filters, pads, shape_strides,
                                                shape_dilations, shape_out_backprop)

    # get placeholder
    dx_shape, dedy, filter_frac, input_sizes, shape_out_backprop = \
        _config_placeholder(shape_out_backprop, shape_filters, input_sizes, filters_dtype,
                            out_backprop_dtype, range_dedy, range_input)

    if groups != 1:
        cube_err.raise_err_specific('conv3d_backprop_input', "group must be 1 now.")

    res = check_conv3dbp_input_params(shape_filters, shape_out_backprop,
                                      input_sizes, shape_strides, pads, groups,
                                      dilations, filters_dtype,
                                      out_backprop_dtype,
                                      res_dtype, kernel_name, range_input, range_dedy)

    (shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
     filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict) = res

    return dx_shape, dedy, filter_frac, shape_filter, shape_out_backprop, input_sizes, strides, \
           pads, dilations, filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict

def _conv3d_backprop_input_compute(filters, out_backprop, y_input, input_size, strides, pads,
                                   dilations=(1, 1, 1, 1, 1), groups=1, data_format="NDHWC",
                                   kernel_name="conv3d_backprop_input"):

    res = check_and_config_para(filters, out_backprop, y_input, input_size, strides, pads,
                                 dilations, groups, data_format, kernel_name)
    (dx_shape, dedy, filter_frac, shape_filter, shape_out_backprop, input_sizes, strides,
     pads, dilations, filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict) = res

    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    shape_filter_ncdhw = [filter_batch, filter_channel, filter_depth, filter_h, filter_w]

    para_dict = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    dedx = conv3d_bp_dx.conv3d_dx(
        filter=filter_frac,
        out_backprop=dedy,
        filter_size=shape_filter_ncdhw,
        input_size=input_sizes,
        para_dict=para_dict
    )

    return {'op_placeholder': [dx_shape, filter_frac, dedy], 'op_res': [dedx]}

@tbe_base.register_operator("Conv3DBackpropInput")
@para_check.check_input_type(dict, dict, dict, dict,
                             (tuple, list), (tuple, list, str),
                             (tuple, list), int, str, str)
def conv3d_backprop_input(input_size, filter, # pylint: disable=R0913,R0914
                          out_backprop, y, strides,
                          pads, dilations=(1, 1, 1, 1, 1), groups=1,
                          data_format="NDHWC",
                          kernel_name="conv3d_backprop_input"):
    """
    algorithm: Conv3d_backprop_input

    Parameters
    ----------
    input_size: dict, will not be used
            input tensor size.

    filter: A dict with keys(shape and dtype)
        Input weight tensor

    out_backprop: A dict with keys(shape and dtype)
        Gradients tensor

    y: A dict with keys(shape and dtype)
        Conv3d_backprop_input output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers
        Filter move stride

    pads: A tuple/list of 6 integers: [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
          str: "SAME" or "VALID"

    dilations: A tuple/list of 5 integers
        filter expand size of dilated conv3d_backprop_input, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
        Default value 1

    data_format: The data format of the input and output data
        Default format "NDHWC"

    kernel_name: Str
        Kernel name, default value is "conv3d_backprop_input"

    Returns
    -------
    None
    """
    with tbe_base.compute():
        res = _conv3d_backprop_input_compute(filter, out_backprop, y, input_size, strides,
                                             pads, dilations, groups, data_format, kernel_name)

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
