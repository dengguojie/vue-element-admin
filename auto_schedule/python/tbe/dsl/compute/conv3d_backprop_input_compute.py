# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
conv3d backprop input DSL interface.
"""
import te.platform as tbe_platform
from tbe.dsl.compute import util as te_util
from tbe.dsl.compute import conv3d_backprop_input_general_compute as conv3d_bp_gen_dx
from tbe.dsl.base.operation import get_te_var
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.errormgr import error_manager_cube as cube_err
from tbe import tvm


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

# the bytes length of several dtype
_BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                   "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
_DATA_SIZE_MAX = 9223372036854775807
_PADDING_VAILD = [0, 0, 0, 0, 0, 0]
# align with 16 for chips
_C0_SIZE = tbe_platform.C0_SIZE
_BLOCK_SIZE = 16


def _get_var_map(out_backprop):
    var_map = {}
    if isinstance(out_backprop.shape[0], tvm.expr.Var):
        var_map["batch_n"] = get_te_var("batch_n").get_tvm_var()
    if isinstance(out_backprop.shape[1], tvm.expr.Var):
        var_map["dedy_d"] = get_te_var("dedy_d").get_tvm_var()
        var_map["dedx_d"] = get_te_var("dedx_d").get_tvm_var()
    if isinstance(out_backprop.shape[3], tvm.expr.Var):
        var_map["dedy_h"] = get_te_var("dedy_h").get_tvm_var()
        var_map["dedx_h"] = get_te_var("dedx_h").get_tvm_var()
    if isinstance(out_backprop.shape[4], tvm.expr.Var):
        var_map["dedy_w"] = get_te_var("dedy_w").get_tvm_var()
        var_map["dedx_w"] = get_te_var("dedx_w").get_tvm_var()
    return var_map


@para_check.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (list, tuple),
                             (list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), str, str, dict, dict)
def _conv3d_backprop_input_compute(filters,  # pylint: disable=R0913,R0914
                                   out_backprop, filter_sizes,
                                   input_sizes, strides, padding,
                                   dilations, res_dtype="float16",
                                   kernel_name="conv3d_backprop_input_cce",
                                   group_dict=None, var_map={}):
    """
    DSL interface of conv3d backprop input

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [D, H, W, N, C]

    input_sizes : shape of dE/dX, [N, D, H, W, C]

    strides : list of strides, [stridebatch,
                                strided, strideh, stridew, stridechannel]

    padding : list of padding, [pad_front, pad_tail,
                                pad_up, pad_down, pad_left, pad_right]

    dilations : [1, 1, 1, 1, 1] by default

    res_dtype : dE/dX data type, "float16" by default

    kernel_name : name of kernel, "conv3d_backprop_input_cce" by default

    group_dict: the information needed for group convolution, None by default

    var_map: the information for dynamic shape, {} by default

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    dx_batch, dx_d, dx_h, dx_w, dx_c = input_sizes
    _, _, config_n0 = tbe_platform.CUBE_MKN[res_dtype]['mac']
    shape_dx = (dx_batch, dx_d, te_util.int_ceil_div(dx_c, config_n0), dx_h, dx_w, config_n0)

    cout_g = group_dict["cout_g"]
    cin1_g = group_dict["cin1_g"]
    real_g = group_dict["real_g"]
    dedy_shape = shape_util.shape_to_list(out_backprop.shape)
    dedy_shape[2] = cout_g // _BLOCK_SIZE
    _, _, filter_d, filter_h, filter_w = filter_sizes
    filter_frac_6d = [cout_g, filter_d, cin1_g, filter_h, filter_w, _BLOCK_SIZE]
    input_sizes[-1] = te_util.align(dx_c, _BLOCK_SIZE)

    if var_map:
        DynamicConv3dBpInputParams.tiling_info_dict = {
            "op_type": "conv3d_backprop_input",
            "a_shape": dedy_shape,
            "b_shape": filter_frac_6d,
            "c_shape": input_sizes,
            "a_dtype": out_backprop.dtype,
            "b_dtype": filters.dtype,
            "c_dtype": res_dtype,
            "mad_dtype": "float32",
            "pad": padding,
            "stride": [strides[1], 1, 1],
            "strideh_expand": strides[2],
            "stridew_expand": strides[3],
            "dilation": [1, dilations[2], dilations[3]],
            "group": real_g,
            "fused_coefficient": [0, 0, 0],
            "bias_flag": False,
            "kernel_name": kernel_name,
            "dynamic_shape_flag": True
        }
        DynamicConv3dBpInputParams.var_map = var_map
    pattc = conv3d_bp_gen_dx.DeConvPattern(filter_sizes, strides=strides,
                                           pad=padding, output_shape=shape_dx,
                                           dilations=dilations,
                                           kernel_name=kernel_name,
                                           group_dict=group_dict,
                                           var_map=var_map)
    dy_col = pattc.generate_a(out_backprop)
    w_col = pattc.generate_b(filters)
    dx_ddr = pattc.generate_c(dy_col, w_col)
    return dx_ddr


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple),
                             (list, tuple), str, str, str, dict)
def _check_conv3dbp_input_params_in_dsl(shape_filter, shape_out_backprop,
                                        input_sizes, strides, pads, dilations,
                                        filter_dtype, out_backprop_dtype,
                                        res_dtype, var_map):
    """
    The params check function of conv3d backprop input in DSL API

    Parameters
    -------------------------
    shape_filter : The shape of filter
        5-D with shape (D, H, W, C, N)

    shape_out_backprop : The shape of gradients
        5-D with shape [N, D, H, W, C]

    input_sizes : The shape of feature map
        5-D with shape [N, D, H, W, C]

    strides : A list/tuple of ints. The stride of the sliding window

    pads : A list/tuple of ints or str

    dilations : An optional list/tuple of ints

    filter_dtype : The dtype of filter data

    out_backprop_dtype : The dtype of gradients data

    res_dtype : The dtype of result(De/Dx) data

    var_map: the dynamic shape para, default is {}

    Returns
    -----------------------
    All transformed params
    """
    def _check_attr_range(attr_name, attr_value, attr_min, attr_max):
        if attr_value < attr_min or attr_value > attr_max:
            cube_err.raise_err_attr_range_invalid("conv3d_backprop_input",
                "[{},{}]".format(attr_min, attr_max),
                attr_name,
                str(attr_value))

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

        aub_dedy_size_min = dedy_w_upper * _BLOCK_SIZE * _BIT_RATIO_DICT["float16"]
        aub_filling_size_min = w_value * _BLOCK_SIZE * _BIT_RATIO_DICT["float16"]
        cub_size_min = _BLOCK_SIZE * _BLOCK_SIZE * _BIT_RATIO_DICT["float16"]
        ub_size = tbe_platform.get_soc_spec("UB_SIZE")

        if (aub_dedy_size_min + aub_filling_size_min + cub_size_min) > ub_size:
            dict_args = {
                'errCode': 'E60119'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_l1_limitation():
        w_value = dedy_w_upper * stride_w
        if dedx_w_upper > _BLOCK_SIZE:
            h_value_max = filter_h_dilation + 1
        elif _BLOCK_SIZE % dedx_w_upper == 0:
            h_value_max = filter_h_dilation + _BLOCK_SIZE // dedx_w_upper - 1
        else:
            h_value_max = filter_h_dilation + _BLOCK_SIZE // dedx_w_upper + 1

        a_l1_size = h_value_max * w_value * ((filter_d_dilation - 2) // stride_d + 2) * _BLOCK_SIZE * 2
        b_l1_size = filter_h_dilation * filter_w_dilation * filter_d_dilation * _BLOCK_SIZE * _BLOCK_SIZE * 2
        l1_size = tbe_platform.get_soc_spec("L1_SIZE")
        if (a_l1_size + b_l1_size) > l1_size:
            dict_args = {
                'errCode': 'E60026'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_shape_error():
        # Check Batch Dimension
        if "batch_n" not in var_map and fmap_batch != dedy_batch:
            cube_err.raise_err_two_paras('E62503', 'conv3d_backprop_input', str(dedy_batch), str(fmap_batch))
        # Check dhw dimension
        if "dedy_h" not in var_map and "dedy_w" not in var_map and "dedy_d" not in var_map:
            pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right = pads
            fmap_h_padding = fmap_h + pad_up + pad_down
            fmap_w_padding = fmap_w + pad_left + pad_right
            fmap_d_padding = fmap_deep + pad_head + pad_tail
            if filter_h_dilation > fmap_h_padding:
                cube_err.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'H',
                                               str(filter_h_dilation), str(fmap_h_padding))
            if filter_w_dilation > fmap_w_padding:
                cube_err.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'W',
                                               str(filter_w_dilation), str(fmap_w_padding))
            if filter_d_dilation > fmap_d_padding:
                cube_err.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'D',
                                               str(filter_d_dilation), str(fmap_d_padding))

            if ((fmap_h_padding - filter_h_dilation) // stride_h + 1) != dedy_h:
                dict_args = {
                    'errCode': 'E60024',
                }
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
            if ((fmap_w_padding - filter_w_dilation) // stride_w + 1) != dedy_w:
                dict_args = {
                    'errCode': 'E60025',
                }
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
            if ((fmap_d_padding - filter_d_dilation) // stride_d + 1) != dedy_deep:
                dict_args = {
                    'errCode': 'E62508',
                }
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))

    dilation_n, dilation_d, dilation_h, dilation_w, dilation_c = dilations
    if dilation_d != 1:
        cube_err.raise_err_specific("conv3d_backprop_input", "dilation in D dimension only supports 1.")

    # dtype check
    filter_dtype = filter_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    para_check.check_dtype_rule(filter_dtype, ('float16'), "filter")
    para_check.check_dtype_rule(out_backprop_dtype, ('float16'), "out_backprop")
    para_check.check_dtype_rule(res_dtype, ('float16'), "output")

    # the relation limits between shape
    shape_filter = list(shape_filter)
    shape_out_backprop = list(shape_out_backprop)
    input_sizes = list(input_sizes)
    strides = list(strides)
    fmap_batch, fmap_deep, fmap_h, fmap_w, fmap_channel = input_sizes
    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel_aligned = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    _, stride_d, stride_h, stride_w, _ = strides

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    filter_d_dilation = (filter_depth - 1) * dilation_d + 1
    pads = list(pads)

    if "dedy_d" in var_map:
        dedy_d_bound = get_te_var("dedy_d").get_bound()
        dedx_d_bound = get_te_var("dedx_d").get_bound()
        dedy_d_upper, dedx_d_upper = dedy_d_bound[1], dedx_d_bound[1]
    else:
        dedy_d_upper, dedx_d_upper = dedy_deep, fmap_deep
    if "dedy_h" in var_map:
        dedy_h_bound = get_te_var("dedy_h").get_bound()
        dedx_h_bound = get_te_var("dedx_h").get_bound()
        dedy_h_upper, dedx_h_upper = dedy_h_bound[1], dedx_h_bound[1]
        dedy_h_lower, dedx_h_lower = dedy_h_bound[0], dedx_h_bound[0]
    else:
        dedy_h_upper, dedx_h_upper = dedy_h, fmap_h
        dedy_h_lower, dedx_h_lower = dedy_h, fmap_h
    if "dedy_w" in var_map:
        dedy_w_bound = get_te_var("dedy_w").get_bound()
        dedx_w_bound = get_te_var("dedx_w").get_bound()
        dedy_w_upper, dedx_w_upper = dedy_w_bound[1], dedx_w_bound[1]
        dedy_w_lower, dedx_w_lower = dedy_w_bound[0], dedx_w_bound[0]
    else:
        dedy_w_upper, dedx_w_upper = dedy_w, fmap_w
        dedy_w_lower, dedx_w_lower = dedy_w, fmap_w
    if "batch_n" in var_map:
        batch_n_bound = get_te_var("batch_n").get_bound()
        dedy_batch_upper, dedx_batch_upper = batch_n_bound[1], batch_n_bound[1]
    else:
        dedy_batch_upper, dedx_batch_upper = dedy_batch, fmap_batch   

    # Chip Design demand fmap_w must larger than 2 when fmap_h != 1
    if dedx_h_upper != 1 and dedx_w_upper == 1:
        cube_err.raise_err_one_para(
            'E62006', 'conv3d_backprop_input', 'Chip Design demand input_size_w must >=2 when input_size_h != 1')

    # filter value limit
    _check_attr_range("filter's H", filter_h, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range("filter's W", filter_w, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range("filter's D", filter_depth, _FILTER_HW_MIN, _FILTER_D_MAX)

    _check_attr_range("filter H*W", filter_h * filter_w, _FILTER_HW_MIN,
                      _FILTER_HW_SIZE)

    _check_attr_range("filter H*W*D", filter_h * filter_w * filter_depth,
                      _FILTER_HW_MIN, _KHWD_COEFF)

    # stride value limit
    _check_attr_range("stride's D", stride_d, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's H", stride_h, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's W", stride_w, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's H*W",
                      stride_h * stride_w, _STRIDE_HW_MIN, _STRIDE_SIZE_MAX)
    _check_attr_range("stride's H*W*D", stride_h * stride_w * stride_d,
                      _STRIDE_HW_MIN, _STRIDE_SIZE_HWD_MAX)

    # dilation value limit
    _check_attr_range("dilation's N", dilation_n, _DILATION_HW_MIN, _DILATION_HW_MIN)
    _check_attr_range("dilation's C", dilation_c, _DILATION_HW_MIN, _DILATION_HW_MIN)
    _check_attr_range("dilation's H", dilation_h, _DILATION_HW_MIN, _DILATION_HW_MAX)
    _check_attr_range("dilation's W", dilation_w, _DILATION_HW_MIN, _DILATION_HW_MAX)

    # Dedy value limit
    def _check_dedy():
        _check_attr_range("Dedy's H after expands", dedy_h_lower * stride_h,
                          _DEDY_HW_MIN, _DEDY_HW_MAX)
        _check_attr_range("Dedy's H after expands", dedy_h_upper * stride_h,
                          _DEDY_HW_MIN, _DEDY_HW_MAX)
        _check_attr_range("Dedy's W after expands", dedy_w_lower * stride_w,
                          _DEDY_HW_MIN, _DEDY_HW_MAX)
        _check_attr_range("Dedy's W after expands", dedy_w_upper * stride_w,
                          _DEDY_HW_MIN, _DEDY_HW_MAX)

    # Fmap value limit
    def _check_dedx():
        _check_attr_range("Fmap's H", dedx_h_lower, _FMAP_HW_MIN, _FMAP_HW_MAX)
        _check_attr_range("Fmap's W", dedx_w_lower, _FMAP_HW_MIN, _FMAP_HW_MAX)
        _check_attr_range("Fmap's H", dedx_h_upper, _FMAP_HW_MIN, _FMAP_HW_MAX)
        _check_attr_range("Fmap's W", dedx_w_upper, _FMAP_HW_MIN, _FMAP_HW_MAX)

    _check_dedy()
    _check_dedx()
    _check_shape_error()

    if stride_h > 1 or stride_w > 1:
        _check_ub_limitation()

    _check_l1_limitation()
    # check shape size, 64 bits limitation
    # ===========================================================
    fmap_size = (dedx_batch_upper * te_util.align(fmap_channel, _C0_SIZE) *
                 dedx_d_upper * dedx_h_upper * dedx_w_upper)
    dedy_size = (dedy_batch_upper * dedy_channel_aligned * dedy_d_upper *
                 dedy_h_upper * dedy_w_upper)
    filter_size = te_util.align(filter_batch, _C0_SIZE) * te_util.align(
        filter_channel, _C0_SIZE) * filter_depth * filter_h * filter_w
    _check_64bits_limitation("input", fmap_size, dtype=res_dtype)
    _check_64bits_limitation("out_backprop", dedy_size,
                             dtype=out_backprop_dtype)
    _check_64bits_limitation("filter", filter_size, dtype=filter_dtype)


@para_check.check_input_type(tvm.tensor.Tensor,
                             tvm.tensor.Tensor,
                             (list, tuple),
                             (list, tuple),
                             dict)
def conv3d_dx(filter,
              out_backprop,
              filter_size,
              input_size,
              para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    filter : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_size : shape of weight, [N, C, D, H, W]

    input_size : shape of dE/dX, [N, D, H, W, C]

    para_dict : dict of parameters
        strides : list of strides, [stridebatch, strided, strideh, stridew, stridechannel]
        pads : list of padding, [pad_front, pad_tail, pad_up, pad_down, pad_left, pad_right]
        dilations : [1, 1, 1, 1, 1] by default
        res_dtype : dE/dX data type, "float16" by default
        kernel_name : conv3d_backprop_input_cce by default
        group_dict : group of parameters

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    strides = para_dict.get("strides")
    pads = para_dict.get("pads")
    dilations = para_dict.get("dilations", [1, 1, 1, 1, 1])
    res_dtype = para_dict.get("res_dtype", "float16")
    kernel_name = para_dict.get("kernel_name", "conv3d_backprop_input_cce")
    filter_dtype = filter.dtype
    out_backprop_dtype = out_backprop.dtype
    group_dict = para_dict.get("group_dict", None)
    
    out_backprop_shape = shape_util.shape_to_list(out_backprop.shape)
    out_backprop_ndhwc = (out_backprop_shape[0],
                          out_backprop_shape[1],
                          out_backprop_shape[3],
                          out_backprop_shape[4],
                          out_backprop_shape[2] * out_backprop_shape[-1])
    filter_dhwcn = (filter_size[2], filter_size[3], filter_size[4],
                    filter_size[1], filter_size[0])

    if group_dict is None:
        group_dict["real_g"] = 1
        group_dict["cout_g"] = out_backprop_ndhwc[-1]
        group_dict["cin1_g"] = te_util.int_ceil_div(filter_dhwcn[-2], _C0_SIZE)
    var_map = _get_var_map(out_backprop)

    _check_conv3dbp_input_params_in_dsl(filter_dhwcn, out_backprop_ndhwc, input_size, strides,
                                        pads, dilations, filter_dtype, out_backprop_dtype,
                                        res_dtype, var_map=var_map)

    return _conv3d_backprop_input_compute(filter,
                                          out_backprop, filter_size,
                                          input_size, strides, pads,
                                          dilations, res_dtype,
                                          kernel_name,
                                          group_dict, var_map=var_map)

class DynamicConv3dBpInputParams: # pylint: disable=R0903
    """
    Dynamic Conv3dBpInput Params
    """

    var_map = {}
    tiling_info_dict = {}
