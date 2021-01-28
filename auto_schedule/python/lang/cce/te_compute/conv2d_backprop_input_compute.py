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
conv2d backprop input DSL interface.
"""
from inspect import currentframe

from te import tvm
from te.lang.cce.te_compute.conv2d_backprop_input_general_compute import DeConvPattern
from te.lang.cce.te_compute.conv2d_backprop_input_opti_compute import (
    DeConvKernelSize1Pattern
)
from te.lang.cce.te_compute.cube_util import check_pad_zero
from te.lang.cce.te_compute.cube_util import shape_to_list
from te.lang.cce.te_compute.cube_util import GroupDictKeys
from te.platform import cce_conf
from te.platform import cce_params
from te.lang.base.operation_impl import get_te_var
from te.tvm.tensor import Tensor
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import error_manager_util

NoneType = type(None)

# shape dim
DY_SHAPE_DIM = 5
FILTER_SHAPE_DIM = 4
FILTER_DIM = 4
DX_SHAPE_DIM = 4
STRIDES_DIM = 2
PADDING_DIM = 4
DILATION_DIM = 4

# padH, padW must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

DY_FILLING_HW_MIN = 2
DY_FILLING_HW_MAX = 4096

# fmapH, fmapW must be in [2,4096]
DX_HW_MIN = 2
DX_HW_MAX = 4096

# conv1d situation support w not larger than 2^31-1
CONV1D_W_MAX = 2147483647

# stride must be in [1,64]
STRIDE_MIN = 1
STRIDE_MAX = 63
STRIDE_MUL_MIN = 1

# the bytes length of several dtype
BIT_RATIO_DICT = {
    "int32": 4,
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "int8": 1,
    "uint4": 0.5,
    "int4": 0.5
}
# same as (2**63-1)
DATA_SIZE_MAX = 9223372036854775807


class DeconvParam:
    """
    class of deconvParam
    """

    def __init__(self):
        pass

    def get_l1_size(self):
        """
        get the l1_size in deconvparam
        """
        return [self.al1_size, self.bl1_size]

    @classmethod
    def set_default(cls):
        """
        set default l1_size in deconvparam
        """
        cls.al1_size = 0
        cls.bl1_size = 0

    al1_size = 0
    bl1_size = 0
    var_map = {}


def _check_variable_range(variable, mini, maxi, name):
    """
    check variable range

    """
    if (not isinstance(variable, int)) or variable < mini or variable > maxi:
        dict_args = dict()
        dict_args["errCode"] = "E65006"
        dict_args["range"] = "[{},{}]".format(mini, maxi)
        dict_args["attr_name"] = name
        dict_args["value"] = str(variable)
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_equal_rule(param_1, param_2, param_name1, param_name2):
    """
    check variable equal

    """
    if param_1 != param_2:
        dict_args = {}
        dict_args["errCode"] = "E65007"
        dict_args["param1"] = param_name1
        dict_args["param2"] = param_name2
        dict_args["actual_value"] = "{}, {}".format(param_1, param_2)
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_input_params(  # pylint: disable=R0913,R0914,R0915
    filters,
    out_backprop,
    filter_sizes,
    input_sizes,
    strides,
    padding,
    dilations,
    res_dtype,
    offset_w,
    group_dict,
    fusion_para=None,
    switch_to_general_scheme=False
):
    """
    check the input params of conv2d_backprop_input_compute

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [N, C, H, W]

    input_sizes : shape of dE/dX, [N, C, H, W]

    strides : list of strides, [strideh, stridew]

    padding : list of padding, [pad_up, pad_down, pad_left, pad_right]

    dilations : [1, 1, 1, 1] by default

    res_dtype : dE/dX data type, "float16" by default

    offset_w: the offset for weight

    group_dict : The params of group convolution.

    fusion_para: the l1 fusion para, default is None

    switch_to_general_scheme: the condition change to general defualt is False

    Returns
    ----------
    None
    """

    def _check_shape_rule(shape_arg, shape_dim, shape_dtype, shape_name):
        if len(shape_arg) != shape_dim:
            dict_args = dict()
            dict_args["errCode"] = "E60006"
            dict_args["param_name"] = shape_name
            dict_args["expected_length"] = str(shape_dim)
            dict_args["length"] = str(len(shape_arg))
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )
        axis_i = 0
        for i in shape_arg:
            if not isinstance(i, shape_dtype):
                dict_args = dict()
                dict_args["errCode"] = "E65001"
                dict_args["param_name"] = shape_name
                dict_args["axis_rule"] = str(shape_dtype)
                dict_args["wrong_axis"] = str(axis_i)
                dict_args["actual_value"] = str(i)
                raise RuntimeError(
                    dict_args, error_manager_util.get_error_message(dict_args)
                )
            axis_i = axis_i + 1

    # check dtype
    def _check_dtype(valid_dtype_dict):
        def _gen_dict_args(name, dtype_list, type_value):
            dict_args = dict()
            dict_args["errCode"] = "E60011"
            dict_args["attr_name"] = name
            dict_args["range"] = str(dtype_list)
            dict_args["value"] = type_value
            return dict_args

        if filters.dtype not in valid_dtype_dict["filter"]:
            dict_args = _gen_dict_args(
                "filter dtype", valid_dtype_dict["filter"], filters.dtype
            )
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )

        if out_backprop.dtype not in valid_dtype_dict["dedy"]:
            dict_args = _gen_dict_args(
                "out_backprop dtype", valid_dtype_dict["dedy"], out_backprop.dtype
            )
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )

        if filters.dtype != out_backprop.dtype:
            dict_args = dict()
            dict_args["errCode"] = "E65002"
            dict_args["param_1"] = "filter"
            dict_args["param_2"] = "out_backprop"
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )

        if res_dtype not in valid_dtype_dict["dx"][filters.dtype]:
            dict_args = _gen_dict_args("dx dtype", valid_dtype_dict["dx"], res_dtype)
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )

    # check shape
    def _check_shape():
        if len(filters.shape) != FILTER_SHAPE_DIM:
            dict_args = dict()
            dict_args["errCode"] = "E65003"
            dict_args["param_name"] = "filter.shape"
            dict_args["format"] = "[k1, n1, n0, k0]"
            dict_args["expect_dim"] = str(FILTER_SHAPE_DIM)
            dict_args["dim"] = str(len(filters.shape))
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )

        if len(out_backprop.shape) != DY_SHAPE_DIM:
            dict_args = dict()
            dict_args["errCode"] = "E65003"
            dict_args["param_name"] = "out_backprop.shape"
            dict_args["format"] = "[No, Co1, Ho, Wo, Co0]"
            dict_args["expect_dim"] = str(DY_SHAPE_DIM)
            dict_args["dim"] = str(len(out_backprop.shape))
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )

        _check_shape_rule(filter_sizes, FILTER_DIM, int, "filter_sizes")
        _check_shape_rule(strides, STRIDES_DIM, int, "strides")
        _check_shape_rule(dilations, DILATION_DIM, int, "dilations")

        if not var_map:
            _check_shape_rule(input_sizes, DX_SHAPE_DIM, int, "input_sizes")
            _check_shape_rule(padding, PADDING_DIM, int, "padding")

    # limitation by chip
    def _is_load3d_special_case():
        # limitation by chip:
        # Ascend910 load3d not support
        # when only fmap w after padding equals to filter w
        if (
            cce_conf.CceProductParams().is_cloud_version()
            and dx_h_after_pad != filter_h
            and dx_w_after_pad == filter_w
        ):
            return False
        return True

    # limitation under conv1d
    def _is_conv1d_situation():
        if dx_h_after_pad == 1 and filter_h_dilation == 1 and stride_h == 1:
            return True
        return False

    g_extend = group_dict.get(GroupDictKeys.g_extend)
    dx_c1_extend = group_dict.get(GroupDictKeys.dx_c1_extend)
    dy_c1_extend = group_dict.get(GroupDictKeys.dy_c1_extend)
    dx_c_ori = group_dict.get(GroupDictKeys.dx_c_ori)
    filter_c_ori = group_dict.get(GroupDictKeys.filter_c_ori)
    groups = group_dict.get(GroupDictKeys.groups)
    filter_ori_format = group_dict.get(GroupDictKeys.filter_ori_format)
    var_map = DeconvParam.var_map

    valid_dtype_dict = {}
    valid_dtype_dict["filter"] = ("float16", "int8")
    valid_dtype_dict["dedy"] = ("float16", "int8")
    valid_dtype_dict["dx"] = {"float16": "float16", "int8": "int32"}

    _check_dtype(valid_dtype_dict)
    _check_shape()

    # begin to fetch params
    if filters.dtype == "int8":
        filter_cout1, _, filter_cin0, filter_cout0 = shape_to_list(filters.shape)
        filter_cout1 = filter_cout1 / filter_sizes[2] / filter_sizes[3] / g_extend
    else:
        _, filter_cout1, filter_cout0, filter_cin0 = shape_to_list(filters.shape)

    dy_batch, dy_c1, dy_h, dy_w, dy_c0 = shape_to_list(out_backprop.shape)
    filter_cout, filter_cin, filter_h, filter_w = filter_sizes
    dx_batch, dx_c, dx_h, dx_w = input_sizes
    stride_h, stride_w = strides
    pad_up, pad_down, pad_left, pad_right = padding
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    dx_h_after_pad = dx_h + pad_up + pad_down
    dx_w_after_pad = dx_w + pad_left + pad_right
    _, dedy_k0, _ = cce_params.CUBE_MKN[out_backprop.dtype]["mac"]
    _, w_k0, w_n0 = cce_params.CUBE_MKN[filters.dtype]["mac"]

    filter_cout = (filter_cout + w_k0 - 1) // w_k0 * w_k0

    # special cases
    dy_filling_hw_min, dx_hw_min = DY_FILLING_HW_MIN, DX_HW_MIN
    dy_filling_hw_max, dx_hw_max = DY_FILLING_HW_MAX, DX_HW_MAX

    if fusion_para and fusion_para.get("valid_shape"):
        dy_h = fusion_para.get("valid_shape")[2]
        dy_filling_hw_min = 1
        dx_hw_min = 1
    # limitation by chip:
    # if kernel h,w in [1,11] and fmap h/w after padding equals to filter h/w
    # load3d support h,w is 1

    if _is_load3d_special_case():
        dy_filling_hw_min = 1
        dx_hw_min = 1
    # if conv1d situation, make sure w is in [1,CONV1D_W_MAX]
    if _is_conv1d_situation():
        dy_filling_hw_min = 1
        dx_hw_min = 1
        dy_filling_hw_max = CONV1D_W_MAX
        dx_hw_max = CONV1D_W_MAX

    if "batch_n" in var_map:
        batch_n_bound = get_te_var("batch_n").get_bound()
    if "dedy_h" in var_map:
        dedy_h_bound = get_te_var("dedy_h").get_bound()
        dx_h_bound = get_te_var("dx_h").get_bound()
    if "dedy_w" in var_map:
        dedy_w_bound = get_te_var("dedy_w").get_bound()
        dx_w_bound = get_te_var("dx_w").get_bound()

    if offset_w is not None:
        dict_args = dict()
        dict_args["errCode"] = "E65004"
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    # dy
    def _check_dy():
        _check_equal_rule(dy_c0, dedy_k0, "dy_c0", str(dedy_k0))
        _check_variable_range(
            dy_h * stride_h, dy_filling_hw_min, dy_filling_hw_max, "dy_h*stride_h"
        )
        if filter_h == 1 and filter_w == 1:
            _check_variable_range(
                dy_w * stride_w * stride_h,
                dy_filling_hw_min,
                dy_filling_hw_max,
                "dy_w*stride_w*stride_h"
            )
        else:
            _check_variable_range(
                dy_w * stride_w,
                dy_filling_hw_min,
                dy_filling_hw_max,
                "dy_w*stride_w"
            )

    # w
    # check filter shape and filter_sizes from topi
    def _check_filter():
        _check_equal_rule(filter_cout0, w_k0, "filter_cout0", str(w_k0))

        _check_equal_rule(filter_cout1, dy_c1_extend,
                          "filter_cout1", "dy_c1_extend")

        _check_equal_rule(filter_cin0, w_n0, "filter_cin0", str(w_n0))

        _check_variable_range(filter_h, FILTER_HW_MIN, FILTER_HW_MAX, "filter_h")

        _check_variable_range(filter_w, FILTER_HW_MIN, FILTER_HW_MAX, "filter_w")

        def _check_max(x_1, x_2, name_1, name_2):
            if x_1 > x_2:
                dict_args = {}
                dict_args["errCode"] = "E65005"
                dict_args["param_1"] = name_1
                dict_args["param_2"] = name_2
                raise RuntimeError(
                    dict_args, error_manager_util.get_error_message(dict_args)
                )

        if "dedy_h" not in var_map:
            _check_max(
                filter_h_dilation,
                dx_h_after_pad,
                "filter_h after dilation",
                "dx_h after pad"
            )
        if "dedy_w" not in var_map:
            _check_max(
                filter_w_dilation,
                dx_w_after_pad,
                "filtre_w after dilation",
                "dx_w after pad"
            )

    # dx
    def _check_dx():
        _check_variable_range(dx_h, dx_hw_min, dx_hw_max, "dx_h")
        _check_variable_range(dx_w, dx_hw_min, dx_hw_max, "dx_w")
        _check_equal_rule(dx_batch, dy_batch, "dx_batch", "dy_batch")

        _check_equal_rule(
            (dx_h_after_pad - filter_h_dilation) // stride_h + 1,
            dy_h,
            "(dx_h_after_pad - filter_h_dilation) \
                // stride_h + 1",
            "dy_h"
        )

        _check_equal_rule(
            (dx_w_after_pad - filter_w_dilation) // stride_w + 1,
            dy_w,
            "(dx_w_after_pad - filter_w_dilation) \
                // stride_w + 1",
            "dy_h"
        )

        _check_equal_rule(dx_c_ori, filter_c_ori, "dx_cin", "filter_cin")
    # strides
    def _check_strides():
        _check_variable_range(stride_h, STRIDE_MIN, STRIDE_MAX, "stride_h")

        _check_variable_range(stride_w, STRIDE_MIN, STRIDE_MAX, "stride_w")

    # padding
    def _check_padding():
        def _check_border(attr_name, attr_value):
            if attr_value < PAD_MIN or attr_value > PAD_MAX:
                dict_args = dict()
                dict_args["errCode"] = "E65006"
                dict_args["range"] = "[{},{}]".format(PAD_MIN, PAD_MAX)
                dict_args["attr_name"] = attr_name
                dict_args["value"] = str(attr_value)
                raise RuntimeError(
                    dict_args, error_manager_util.get_error_message(dict_args)
                )

        if fusion_para.get("l1_fusion_type") == -1:
            _check_border("pad_up", pad_up)
            _check_border("pad_down", pad_down)

            _check_border("pad_left", pad_left)
            _check_border("pad_right", pad_right)

    # dilation
    def _check_dilation():
        if dilation_n != 1 or dilation_c != 1:
            dict_args = dict()
            dict_args["errCode"] = "E60023"
            dict_args["dilation_n"] = str(dilation_n)
            dict_args["dilation_c"] = str(dilation_c)
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )

        _check_variable_range(dilation_h, DILATION_MIN, DILATION_MAX, "dilation_h")

        _check_variable_range(dilation_w, DILATION_MIN, DILATION_MAX, "dilation_w")

    # check L1 exceed buffer
    def _check_l1_buffer():
        def _l1fusion_size_limit(l1_size):
            l1fusion_l1_size = 0
            if (
                padding != [0, 0, 0, 0]
                or [filter_h, filter_w] != [1, 1]
                or switch_to_general_scheme
            ):
                if stride_h > 1 or stride_w > 1:
                    l1fusion_l1_size = l1_size
            return l1fusion_l1_size

        if "dedy_w" in var_map:
            dx_w = dx_w_bound[1]
            dy_w = dedy_w_bound[1]
        else:
            dx_w = input_sizes[3]
            dy_w = shape_to_list(out_backprop.shape)[3]
        if not dy_w:
            return
        c0_size_k = cce_params.CUBE_MKN[filters.dtype]["mac"][1]
        bl1_size = (
            filter_h_dilation
            * filter_w_dilation
            * cce_params.C0_SIZE
            * c0_size_k
            * BIT_RATIO_DICT.get(filters.dtype)
        )

        al1_w_value = dy_w * stride_w

        if dx_w > cce_params.C0_SIZE:
            al1_h_value = filter_h_dilation + 1
        elif cce_params.C0_SIZE % dx_w == 0:
            al1_h_value = filter_h_dilation + cce_params.C0_SIZE // dx_w - 1
        else:
            al1_h_value = filter_h_dilation + cce_params.C0_SIZE // dx_w + 1

        al1_size = (
            al1_h_value
            * al1_w_value
            * c0_size_k
            * BIT_RATIO_DICT.get(out_backprop.dtype)
        )
        if _is_conv1d_situation():
            load3d_stride = 1
            a_l1_m_length = (dy_c0 - 1) * load3d_stride + filter_w_dilation
            al1_size = (
                a_l1_m_length * c0_size_k * BIT_RATIO_DICT.get(out_backprop.dtype)
            )
        if fusion_para.get("l1_fusion_type") != -1:
            al1_size = _l1fusion_size_limit(al1_size)
        DeconvParam.al1_size = al1_size
        DeconvParam.bl1_size = bl1_size
        if al1_size + bl1_size > cce_conf.get_soc_spec("L1_SIZE"):
            dict_args = dict()
            dict_args["errCode"] = "E60026"
            raise RuntimeError(
                dict_args, error_manager_util.get_error_message(dict_args)
            )

    # 64 bits limitation check
    def _check_chip_limitation():
        def _align(x_1, x_2):
            return (x_1 + x_2 - 1) // x_2 * x_2

        def _check_64bits_limitation(attr_value, dtype=None):
            if dtype is None:
                bit_ratio = BIT_RATIO_DICT.get("float16")
            else:
                bit_ratio = BIT_RATIO_DICT.get(dtype)
            if attr_value * bit_ratio > DATA_SIZE_MAX:
                dict_args = dict()
                dict_args["errCode"] = "E60020"
                raise RuntimeError(
                    dict_args, error_manager_util.get_error_message(dict_args)
                )

        _, dedy_k0, _ = cce_params.CUBE_MKN[out_backprop.dtype]["mac"]
        _, w_k0, w_n0 = cce_params.CUBE_MKN[filters.dtype]["mac"]

        if "batch_n" in var_map:
            dy_batch_upper, dx_batch_upper = batch_n_bound[1], batch_n_bound[1]
        else:
            dy_batch_upper, dx_batch_upper = dy_batch, dx_batch
        if "dedy_h" in var_map:
            dy_h_upper, dx_h_upper = dedy_h_bound[1], dx_h_bound[1]
        else:
            dy_h_upper, dx_h_upper = dy_h, dx_h
        if "dedy_w" in var_map:
            dy_w_upper, dx_w_upper = dedy_w_bound[1], dx_w_bound[1]
        else:
            dy_w_upper, dx_w_upper = dy_w, dx_w
        if dy_batch_upper and dy_h_upper and dy_w_upper:
            fmap_size = dx_batch_upper * _align(dx_c, dedy_k0) * dx_h_upper * dx_w_upper
            dedy_size = dy_batch_upper * dy_c1 * dy_h_upper * dy_w_upper * dy_c0
            _check_64bits_limitation(fmap_size, dtype=res_dtype)
            _check_64bits_limitation(dedy_size, dtype=out_backprop.dtype)
        filter_size = (
                          _align(dy_c1_extend, w_k0) * _align(dx_c1_extend, w_n0)
                      ) * filter_h * filter_w * g_extend
        _check_64bits_limitation(filter_size, dtype=filters.dtype)

    if "dx_h" not in var_map and "dx_w" not in var_map:
        _check_dy()
        _check_dx()
        _check_padding()

    _check_filter()
    _check_strides()
    _check_dilation()
    _check_l1_buffer()
    _check_chip_limitation()


def _get_var_map(out_backprop):
    var_map = {}
    if isinstance(out_backprop.shape[0], tvm.expr.Var):
        var_map["batch_n"] = get_te_var("batch_n").get_tvm_var()
    if isinstance(out_backprop.shape[2], tvm.expr.Var):
        var_map["dedy_h"] = get_te_var("dedy_h").get_tvm_var()
        var_map["dx_h"] = get_te_var("dx_h").get_tvm_var()
    if isinstance(out_backprop.shape[3], tvm.expr.Var):
        var_map["dedy_w"] = get_te_var("dedy_w").get_tvm_var()
        var_map["dx_w"] = get_te_var("dx_w").get_tvm_var()
    return var_map


@para_check.check_input_type(Tensor, Tensor, (list, tuple), (list, tuple), dict)
def conv2d_backprop_input_compute(filters, out_backprop, filter_sizes, input_sizes, para_dict):
    """
    DSL interface of conv2d backprop input

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [N, C, H, W]

    input_sizes : shape of dE/dX, [N, C, H, W]

    para_dict:

        strides : list of strides, [strideh, stridew]

        padding : list of padding, [pad_up, pad_down, pad_left, pad_right]

        dilations : list of dilations, [dilation_n, dilation_c, dilation_h, dilation_w]

        res_dtype : dE/dX data type, "float16" by default

        offset_x : offset of x

        offset_w : offset of w

        fusion_para: the l1 fuison para

        kernel_name : cce kernel name

        group_dict : The params of group convolution.

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """

    strides = para_dict.get("strides")
    padding = para_dict.get("padding")
    dilations = para_dict.get("dilations")
    res_dtype = para_dict.get("res_dtype", "float16")
    tensor_bias = para_dict.get("tensor_bias")
    offset_x = para_dict.get("offset_x", 0)
    offset_w = para_dict.get("offset_w")
    fusion_para = para_dict.get("fusion_para")
    kernel_name = para_dict.get("kernel_name", "conv2d_backprop_input_cce")
    group_dict = para_dict.get("group_dict")

    def ceil(lhs, rhs):
        return (lhs + rhs - 1) // rhs

    DeconvParam.set_default()
    if fusion_para is None:
        fusion_para = {
            "input_memory_type": 0,
            "output_memory_type": 0,
            "valid_shape": (),
            "slice_offset": (),
            "l1_fusion_type": -1,
            "fmap_l1_addr_flag": False,
            "fmap_l1_valid_size": 0
        }

    if group_dict is None:
        group_dict = {GroupDictKeys.g_extend: 1,
                      GroupDictKeys.multiple_extend: 1,
                      GroupDictKeys.groups: 1,
                      GroupDictKeys.dx_c1_extend: ceil(input_sizes[1],
                                                       cce_params.C0_SIZE),
                      GroupDictKeys.dy_c1_extend: shape_to_list(out_backprop.shape)[1],
                      GroupDictKeys.dx_c_ori: input_sizes[1],
                      GroupDictKeys.dy_c_ori: filter_sizes[0],
                      GroupDictKeys.filter_batch_ori: filter_sizes[0],
                      GroupDictKeys.filter_c_ori: filter_sizes[1],
                      GroupDictKeys.filter_ori_format: "NCHW"
                      }

    caller_name = currentframe().f_back.f_back.f_code.co_name

    # opti -> general:
    # 1) In quantified non-fusion scene, opti strategy maybe exceed UB size;
    # 2) In quantified fusion scene, convert to general strategy for increasing
    # bias in l0c
    def _is_switch_to_general_scheme():
        if (
            tensor_bias is not None
            and filters.dtype == "int8"
            and out_backprop.dtype == "int8"
            and (strides[0] > 1 or strides[1] > 1)
        ):
            if caller_name.endswith("_compute") or (
                1 + strides[0] * strides[1]
            ) * shape_to_list(out_backprop.shape)[3] * cce_params.CUBE_MKN[res_dtype][
                "mac"
            ][
                2
            ] * BIT_RATIO_DICT[
                res_dtype
            ] > cce_conf.get_soc_spec(
                "UB_SIZE"
            ):
                return True
        return False

    DeconvParam.var_map = _get_var_map(out_backprop)

    switch_to_general_scheme = _is_switch_to_general_scheme()

    _check_input_params(
        filters,
        out_backprop,
        filter_sizes,
        input_sizes,
        strides,
        padding,
        dilations,
        res_dtype,
        offset_w,
        group_dict,
        fusion_para=fusion_para,
        switch_to_general_scheme=switch_to_general_scheme
    )

    out_channel, in_channel, filter_h, filter_w = filter_sizes
    dx_batch, dx_c, dx_h, dx_w = input_sizes

    _, dx_k0, dx_n0 = cce_params.CUBE_MKN[res_dtype]["mac"]
    shape_dx = (dx_batch, ceil(dx_c, dx_n0), dx_h, dx_w, dx_n0)

    shape_dy = shape_to_list(out_backprop.shape)
    g_extend = group_dict.get(GroupDictKeys.g_extend)
    dy_c1_extend = group_dict.get(GroupDictKeys.dy_c1_extend)
    dx_c1_extend = group_dict.get(GroupDictKeys.dx_c1_extend)
    dy_6gd_shape = [g_extend, shape_dy[0], dy_c1_extend] + shape_dy[2:]
    dx_6gd_shape = [g_extend, shape_dx[0], dx_c1_extend] + list(shape_dx)[2:]

    DynamicConv2dBpInputParams.tiling_info_dict = {
        "op_type": "conv2d_backprop_input",
        "A_shape": dy_6gd_shape[1:],
        "B_shape": [dy_c1_extend * dx_k0,
                    dx_c1_extend,
                    filter_h, filter_w, dx_n0],
        "C_shape": dx_6gd_shape[1:],
        "A_dtype": out_backprop.dtype,
        "B_dtype": filters.dtype,
        "C_dtype": res_dtype,
        "mad_dtype": "float32",
        "padl": padding[2],
        "padr": padding[3],
        "padu": padding[0],
        "padd": padding[1],
        "strideH": 1,
        "strideW": 1,
        "strideH_expand": strides[0],
        "strideW_expand": strides[1],
        "dilationH": dilations[2],
        "dilationW": dilations[3],
        "group": group_dict.get(GroupDictKeys.g_extend),
        "bias_flag": False,
        "fused_double_operand_num": 0,
        "in_fm_memory_type": [],
        "out_fm_memory_type": [],
        "l1_fusion_type": -1,
        "fusion_type": 1,
        "kernel_name": kernel_name,
        "dynamic_shape_flag": True
    }
    DynamicConv2dBpInputParams.var_map = DeconvParam.var_map

    if (
        filter_h == 1
        and filter_w == 1
        and check_pad_zero(padding)
        and not switch_to_general_scheme
    ):
        pattc = DeConvKernelSize1Pattern(
            filter_sizes,
            strides=strides,
            pad=padding,
            output_shape=shape_dx,
            fusion_para=fusion_para,
            kernel_name=kernel_name,
            offset_x=offset_x,
            group_dict=group_dict,
            var_map=DeconvParam.var_map,
        )
    else:
        pattc = DeConvPattern(
            filter_sizes,
            strides=strides,
            pad=padding,
            output_shape=shape_dx,
            dilations=dilations,
            offset_x=offset_x,
            fusion_para=fusion_para,
            kernel_name=kernel_name,
            group_dict=group_dict,
            var_map=DeconvParam.var_map,
        )

    dy_col = pattc.generate_a(out_backprop)
    w_col = pattc.generate_b(filters)
    dx_ddr = pattc.generate_c(dy_col, w_col, tensor_bias=tensor_bias)

    return dx_ddr


class DynamicConv2dBpInputParams:  # pylint: disable=R0903
    """
    Dynamic Conv2dBpInput Params
    """

    def __init__(self):
        pass

    tiling_info_dict = {}
    var_map = {}
