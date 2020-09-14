"""
conv2d backprop input DSL interface.
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
import inspect
from te import tvm
from te.platform import CUBE_MKN
from te.platform import get_soc_spec
from te.platform import cce_params
from te.platform import operation
from te.platform.cce_conf import CceProductParams as pver
from te.lang.cce.te_compute import util
from te.lang.cce.te_compute.cube_util import shape_to_list
from te.utils.error_manager import error_manager_util as err_man
from .conv2d_backprop_input_general_compute import DeConvPattern
from .conv2d_backprop_input_opti_compute import DeConvKernelSize1Pattern

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
STRIDE_MUL_MAX = 256

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
# same as (2**63-1)
DATA_SIZE_MAX = 9223372036854775807

# for dynamic shape tiling
DYNAMIC_TILING_INFO_DICT = {}

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
        cls.al1_size = 0
        cls.bl1_size = 0

    al1_size = 0
    bl1_size = 0


def check_variable_range(variable, mini, maxi, name):
    """
    check variable range

    """
    if (not isinstance(variable, int)) or variable < mini \
            or variable > maxi:
        dict_args = dict()
        dict_args['errCode'] = "E65006"
        dict_args['range'] = "[{},{}]".format(mini, maxi)
        dict_args['attr_name'] = name
        dict_args["value"] = str(variable)
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))


def check_equal_rule(param_1, param_2, param_name1, param_name2):
    """
    check variable equal

    """
    if param_1 != param_2:
        dict_args = {}
        dict_args['errCode'] = "E65007"
        dict_args['param1'] = param_name1
        dict_args['param2'] = param_name2
        dict_args['actual_value'] = "{}, {}".format(param_1, param_2)
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))

def check_input_params(filters, # pylint: disable=R0913,R0914,R0915
                       out_backprop, filter_sizes, input_sizes,
                       strides, padding, dilations, res_dtype, offset_w,
                       fusion_para=None, dynamic_para=None):
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
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))
        axis_i = 0
        for i in shape_arg:
            if not isinstance(i, shape_dtype):
                dict_args = dict()
                dict_args["errCode"] = "E65001"
                dict_args["param_name"] = shape_name
                dict_args["axis_rule"] = str(shape_dtype)
                dict_args["wrong_axis"] = str(axis_i)
                dict_args["actual_value"] = str(i)
                raise RuntimeError(dict_args,
                                   err_man.get_error_message(dict_args))
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
            dict_args = \
                _gen_dict_args("filter dtype",
                               valid_dtype_dict["filter"], filters.dtype)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if out_backprop.dtype not in valid_dtype_dict["dedy"]:
            dict_args = \
                _gen_dict_args("out_backprop dtype",
                               valid_dtype_dict["dedy"], out_backprop.dtype)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if filters.dtype != out_backprop.dtype:
            dict_args = dict()
            dict_args["errCode"] = "E65002"
            dict_args["param_1"] = "filter"
            dict_args["param_2"] = "out_backprop"
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if res_dtype not in valid_dtype_dict["dx"][filters.dtype]:
            dict_args = \
                _gen_dict_args("dx dtype",
                               valid_dtype_dict["dx"], res_dtype)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

    # check shape
    def _check_shape():
        if len(filters.shape) != FILTER_SHAPE_DIM:
            dict_args = dict()
            dict_args["errCode"] = "E65003"
            dict_args["param_name"] = "filter.shape"
            dict_args["format"] = "[k1, n1, n0, k0]"
            dict_args["expect_dim"] = str(FILTER_SHAPE_DIM)
            dict_args["dim"] = str(len(filters.shape))
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if len(out_backprop.shape) != DY_SHAPE_DIM:
            dict_args = dict()
            dict_args["errCode"] = "E65003"
            dict_args["param_name"] = "out_backprop.shape"
            dict_args["format"] = "[No, Co1, Ho, Wo, Co0]"
            dict_args["expect_dim"] = str(DY_SHAPE_DIM)
            dict_args["dim"] = str(len(out_backprop.shape))
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        _check_shape_rule(filter_sizes, FILTER_DIM, int, "filter_sizes")
        _check_shape_rule(strides, STRIDES_DIM, int, "strides")
        _check_shape_rule(dilations, DILATION_DIM, int, "dilations")

        if not dynamic_para:
            _check_shape_rule(input_sizes, DX_SHAPE_DIM, int, "input_sizes")
            _check_shape_rule(padding, PADDING_DIM, int, "padding")

    # limitation by chip
    def _is_load3d_special_case():
        if pver().is_cloud_version() \
            and dx_h_after_pad != filter_h \
            and dx_w_after_pad == filter_w:
            return False
        if (1 <= filter_h <= 11) and (1 <= filter_w <= 11) \
            and (dx_h_after_pad == filter_h or dx_w_after_pad == filter_w):
            return True
        return False

    # limitation under conv1d
    def _is_conv1d_situation():
        if dx_h_after_pad == 1 and filter_h_dilation == 1 and stride_h == 1:
            return True
        return False

    valid_dtype_dict = {}
    valid_dtype_dict["filter"] = ("float16", "int8")
    valid_dtype_dict["dedy"] = ("float16", "int8")
    valid_dtype_dict["dx"] = {"float16": "float16", "int8": "int32"}

    _check_dtype(valid_dtype_dict)
    _check_shape()

    # begin to fetch params
    if filters.dtype == "int8":
        filter_cout1, _, filter_cin0, filter_cout0 \
            = shape_to_list(filters.shape)
        filter_cout1 = filter_cout1 / filter_sizes[2] / filter_sizes[3]
    else:
        _, filter_cout1, filter_cout0, filter_cin0 \
            = shape_to_list(filters.shape)

    dy_batch, dy_c1, dy_h, dy_w, dy_c0 = \
        shape_to_list(out_backprop.shape)
    filter_cout, filter_cin, filter_h, filter_w = filter_sizes
    dx_batch, dx_c, dx_h, dx_w = input_sizes
    stride_h, stride_w = strides
    pad_up, pad_down, pad_left, pad_right = padding
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    dx_h_after_pad = dx_h + pad_up + pad_down
    dx_w_after_pad = dx_w + pad_left + pad_right
    _, dedy_k0, _ = CUBE_MKN[out_backprop.dtype]['mac']
    _, w_k0, w_n0 = CUBE_MKN[filters.dtype]['mac']

    filter_cout = (filter_cout + w_k0 - 1) // w_k0 * w_k0
    dynamic_mode = None if not dynamic_para \
        else dynamic_para.get("dynamic_mode")
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

    if dynamic_mode == "dynamic_hw":
        dy_filling_hw_min = 1
        dx_hw_min = 1
        dedy_h_bound = operation.get_te_var("dedy_h").get_bound()
        dedy_w_bound = operation.get_te_var("dedy_w").get_bound()
        dx_h_bound = operation.get_te_var("dx_h").get_bound()
        dx_w_bound = operation.get_te_var("dx_w").get_bound()
    elif dynamic_mode == "dynamic_batch":
        batch_n_bound = operation.get_te_var("batch_n").get_bound()

    if offset_w is not None:
        dict_args = dict()
        dict_args["errCode"] = "E65004"
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))

    # dy
    def _check_dy():
        check_equal_rule(dy_c0, dedy_k0, "dy_c0", str(dedy_k0))
        if dynamic_mode == "dynamic_hw":
            check_variable_range(dedy_h_bound[0]*stride_h, dy_filling_hw_min,
                                 dy_filling_hw_max, "dy_h*stride_h")
            check_variable_range(dedy_h_bound[1]*stride_h, dy_filling_hw_min,
                                 dy_filling_hw_max, "dy_h*stride_h")
            if filter_h == 1 and filter_w == 1:
                check_variable_range(dedy_w_bound[0]*stride_w*stride_h, 
                                     dy_filling_hw_min,
                                     dy_filling_hw_max, 
                                     "dy_w*stride_w*stride_h")
                check_variable_range(dedy_w_bound[1]*stride_w*stride_h, 
                                     dy_filling_hw_min,
                                     dy_filling_hw_max, 
                                     "dy_w*stride_w*stride_h")
            else:
                check_variable_range(dedy_w_bound[0]*stride_w, 
                                     dy_filling_hw_min,
                                     dy_filling_hw_max, 
                                     "dy_w*stride_w")
                check_variable_range(dedy_w_bound[1]*stride_w, 
                                     dy_filling_hw_min,
                                     dy_filling_hw_max, 
                                     "dy_w*stride_w")
        else:
            check_variable_range(dy_h*stride_h, dy_filling_hw_min,
                                 dy_filling_hw_max, "dy_h*stride_h")
            if filter_h == 1 and filter_w == 1:
                check_variable_range(dy_w*stride_w*stride_h,
                                     dy_filling_hw_min,
                                     dy_filling_hw_max, 
                                     "dy_w*stride_w*stride_h")
            else:
                check_variable_range(dy_w*stride_w, dy_filling_hw_min,
                                    dy_filling_hw_max, "dy_w*stride_w")

    # w
    # check filter shape and filter_sizes from topi
    def _check_filter():
        check_equal_rule(filter_cout0, w_k0, "filter_cout0", str(w_k0))

        check_equal_rule(filter_cout1*filter_cout0, filter_cout,
                         "filter_cout1*filter_cout0", "filter_cout")

        check_equal_rule(filter_cin0, w_n0, "filter_cin0", str(w_n0))

        check_equal_rule(dy_c1*dy_c0, filter_cout, "dy_c", "filter_cout")

        check_variable_range(filter_h, FILTER_HW_MIN,
                             FILTER_HW_MAX, "filter_h")

        check_variable_range(filter_w, FILTER_HW_MIN,
                             FILTER_HW_MAX, "filter_w")

        def _check_max(x_1, x_2, name_1, name_2):
            if x_1 > x_2:
                dict_args = {}
                dict_args["errCode"] = "E65005"
                dict_args["param_1"] = name_1
                dict_args["param_2"] = name_2
                raise RuntimeError(dict_args,
                                   err_man.get_error_message(dict_args))
            if not dynamic_para:
                if not fusion_para.get("valid_shape"):
                    _check_max(filter_h_dilation, dx_h_after_pad,
                            "filter_h after dilation", "dx_h after pad")
                _check_max(filter_w_dilation, dx_w_after_pad,
                        "filtre_w after dilation", "dx_w after pad")

    # dx
    def _check_dx():
        if dynamic_mode == "dynamic_hw":
            check_variable_range(dx_h_bound[0], dx_hw_min, dx_hw_max, "dx_h")
            check_variable_range(dx_h_bound[1], dx_hw_min, dx_hw_max, "dx_h")
            check_variable_range(dx_w_bound[0], dx_hw_min, dx_hw_max, "dx_w")
            check_variable_range(dx_w_bound[1], dx_hw_min, dx_hw_max, "dx_w")
            check_equal_rule(dx_batch, dy_batch, "dx_batch", "dy_batch")
        elif dynamic_mode == "dynamic_batch":
            check_variable_range(dx_h, dx_hw_min, dx_hw_max, "dx_h")
            check_variable_range(dx_w, dx_hw_min, dx_hw_max, "dx_w")
        else:
            check_variable_range(dx_h, dx_hw_min, dx_hw_max, "dx_h")
            check_variable_range(dx_w, dx_hw_min, dx_hw_max, "dx_w")
            check_equal_rule(dx_batch, dy_batch, "dx_batch", "dy_batch")
            if not fusion_para or not fusion_para.get("valid_shape"):
                check_equal_rule(
                    (dx_h_after_pad - filter_h_dilation) \
                        // stride_h + 1, dy_h,
                    "(dx_h_after_pad - filter_h_dilation) \
                        // stride_h + 1", "dy_h")

                check_equal_rule(
                    (dx_w_after_pad - filter_w_dilation) \
                        // stride_w + 1, dy_w,
                    "(dx_w_after_pad - filter_w_dilation) \
                        // stride_w + 1", "dy_h")

        check_equal_rule(dx_c, filter_cin, "dx_cin", "filter_cin")

    # strides
    def _check_strides():
        check_variable_range(stride_h, STRIDE_MIN, STRIDE_MAX, "stride_h")

        check_variable_range(stride_w, STRIDE_MIN, STRIDE_MAX, "stride_w")

        check_variable_range(stride_h*stride_w, STRIDE_MUL_MIN,
                             STRIDE_MUL_MAX, "stride_h*stride_w")

    # padding
    def _check_padding():
        def _check_border(attr_name, attr_value):
            if attr_value < PAD_MIN or attr_value > PAD_MAX:
                dict_args = dict()
                dict_args['errCode'] = "E65006"
                dict_args['range'] = "[{},{}]".format(PAD_MIN, PAD_MAX)
                dict_args['attr_name'] = attr_name
                dict_args["value"] = str(attr_value)
                raise RuntimeError(dict_args,
                                   err_man.get_error_message(dict_args))

        def _check_max(x_1, x_2, name_1, name_2):
            if x_1 > x_2:
                dict_args = {}
                dict_args["errCode"] = "E65005"
                dict_args["param_1"] = name_1
                dict_args["param_2"] = name_2
                raise RuntimeError(dict_args,
                                   err_man.get_error_message(dict_args))

        _check_border("pad_up", pad_up)
        _check_border("pad_down", pad_down)
        _check_border("pad_left", pad_left)
        _check_border("pad_right", pad_right)

        _check_max(pad_up, filter_h_dilation,
                   "pad_up", "filter_h_dilation")
        _check_max(pad_down, filter_h_dilation,
                   "pad_down", "filter_h_dilation")
        _check_max(pad_left, filter_w_dilation,
                   "pad_left", "filter_w_dilation")
        _check_max(pad_right, filter_w_dilation,
                   "pad_right", "filter_w_dilation")

    # dilation
    def _check_dilation():
        if dilation_n != 1 or dilation_c != 1:
            dict_args = dict()
            dict_args["errCode"] = "E60023"
            dict_args["dilation_n"] = str(dilation_n)
            dict_args["dilation_c"] = str(dilation_c)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        check_variable_range(dilation_h, DILATION_MIN,
                             DILATION_MAX, "dilation_h")

        check_variable_range(dilation_w, DILATION_MIN,
                             DILATION_MAX, "dilation_w")

    # check L1 exceed buffer
    def _check_l1_buffer():
        if dynamic_mode == "dynamic_hw":
            dx_w = dx_w_bound[1]
            dy_w = dedy_w_bound[1]
        else:
            dx_w = input_sizes[3]
            dy_w = shape_to_list(out_backprop.shape)[3]
        c0_size = cce_params.C0_SIZE
        c0_size_k = cce_params.CUBE_MKN[filters.dtype]['mac'][1]
        bl1_size = filter_h_dilation * filter_w_dilation *\
                   c0_size * c0_size_k * BIT_RATIO_DICT.get(filters.dtype)

        al1_w_value = dy_w * stride_w

        if dx_w > c0_size:
            al1_h_value = filter_h_dilation + 1
        elif c0_size % dx_w == 0:
            al1_h_value = filter_h_dilation + c0_size // dx_w - 1
        else:
            al1_h_value = filter_h_dilation + c0_size // dx_w + 1

        al1_size = al1_h_value * al1_w_value *\
                   c0_size_k * BIT_RATIO_DICT.get(out_backprop.dtype)
        if _is_conv1d_situation():
            load3d_stride = 1
            a_l1_m_length = (dy_c0 - 1) * load3d_stride + filter_w_dilation
            al1_size = a_l1_m_length *\
                       c0_size_k * BIT_RATIO_DICT.get(out_backprop.dtype)
        DeconvParam.al1_size = al1_size
        DeconvParam.bl1_size = bl1_size
        if al1_size + bl1_size > get_soc_spec("L1_SIZE"):
            dict_args = dict()
            dict_args["errCode"] = "E60026"
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

    # 64 bits limitation check
    def _check_chip_limitation():
        def _align(x_1, x_2):
            return (x_1 + x_2 - 1) // x_2 * x_2

        def _check_64bits_limitation(attr_name, attr_value, dtype=None):
            if dtype is None:
                bit_ratio = BIT_RATIO_DICT.get("float16")
            else:
                bit_ratio = BIT_RATIO_DICT.get(dtype)
            if attr_value*bit_ratio > DATA_SIZE_MAX:
                dict_args = dict()
                dict_args["errCode"] = "E60020"
                raise RuntimeError(dict_args,
                                   err_man.get_error_message(dict_args))

        _, dedy_k0, _ = CUBE_MKN[out_backprop.dtype]['mac']
        _, w_k0, w_n0 = CUBE_MKN[filters.dtype]['mac']

        if dynamic_mode == "dynamic_hw":
            fmap_size = dx_batch * _align(dx_c, dedy_k0) \
                * dx_h_bound[1] * dx_w_bound[1]
            dedy_size = dy_batch * dy_c1 * dedy_h_bound[1] \
                * dedy_w_bound[1] * dy_c0
        elif dynamic_mode == "dynamic_batch":
            fmap_size = batch_n_bound[1] * _align(dx_c, dedy_k0) * dx_h * dx_w
            dedy_size = batch_n_bound[1] * dy_c1 * dy_h * dy_w * dy_c0
        else:
            fmap_size = dx_batch * _align(dx_c, dedy_k0) * dx_h * dx_w
            dedy_size = dy_batch * dy_c1 * dy_h * dy_w * dy_c0
        filter_size = _align(filter_cout, w_k0) \
                      * _align(filter_cin, w_n0) * filter_h * filter_w
        _check_64bits_limitation("fmap_size",
                                 fmap_size, dtype=res_dtype)
        _check_64bits_limitation("dedy_size",
                                 dedy_size, dtype=out_backprop.dtype)
        _check_64bits_limitation("filter_size",
                                 filter_size, dtype=filters.dtype)

    _check_dy()
    _check_filter()
    _check_dx()
    _check_strides()
    if dynamic_mode != "dynamic_hw":
        _check_padding()
    _check_dilation()
    _check_l1_buffer()
    _check_chip_limitation()


@util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor,
                       (list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (list, tuple), str,
                       (tvm.tensor.Tensor, NoneType), int,
                       (tvm.tensor.Tensor, NoneType),
                       (dict, NoneType), (dict, NoneType), str)
def conv2d_backprop_input_compute(filters,# pylint: disable=R0913,R0914
                                  out_backprop, filter_sizes,
                                  input_sizes, strides, padding, dilations,
                                  res_dtype="float16", tensor_bias=None,
                                  offset_x=0, offset_w=None,
                                  fusion_para=None,
                                  dynamic_para=None,
                                  kernel_name="conv2d_backprop_input_cce"):

    """
    DSL interface of conv2d backprop input

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

    offset_x : offset of x

    offset_w : offset of w

    kernel_name : cce kernel name

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """

    DeconvParam.set_default()
    if fusion_para is None:
        fusion_para = {"input_memory_type": 0,
                       "output_memory_type": 0,
                       "valid_shape": (),
                       "slice_offset": (),
                       "output_offset": (),
                       "l1_fusion_type": -1,
                       "fmap_l1_addr_flag": False,
                       "fmap_l1_valid_size": 0}

    check_input_params(filters, out_backprop, filter_sizes, input_sizes,
                        strides, padding, dilations, res_dtype, offset_w,
                        fusion_para=fusion_para,
                        dynamic_para=dynamic_para)

    out_channel, in_channel, filter_h, filter_w = filter_sizes
    dx_batch, dx_c, dx_h, dx_w = input_sizes
    ceil = lambda A, B: (A + B - 1) // B
    _, dx_k0, dx_n0 = CUBE_MKN[res_dtype]['mac']
    shape_dx = (dx_batch, ceil(dx_c, dx_n0), dx_h, dx_w, dx_n0)

    caller_name = inspect.currentframe().f_back.f_back.f_code.co_name
    switch_to_general_scheme = tensor_bias is not None and \
        filters.dtype == 'int8' and out_backprop.dtype == 'int8' and \
        (strides[0] > 1 or strides[1] > 1) and \
        caller_name.endswith('_compute')

    DynamicConv2dBpInputParams.tiling_info_dict = {
        "op_type": 'conv2d_backprop_input',
        "A_shape": shape_to_list(out_backprop.shape),
        "B_shape": [out_channel, (in_channel + dx_k0 - 1) // dx_k0, 
                    filter_h, filter_w, dx_k0],
        "C_shape": list(shape_dx),
        "A_dtype": out_backprop.dtype,
        "B_dtype": filters.dtype,
        "C_dtype": res_dtype,
        "mad_dtype": 'float32',
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
        "group": 1,
        "bias_flag": False,
        "fused_double_operand_num": 0,
        "in_fm_memory_type": [],
        "out_fm_memory_type": [],
        "l1_fusion_type": -1,
        "fusion_type": 0,
        "kernel_name": kernel_name
    }

    if filter_h == 1 and filter_w == 1 \
          and (padding == [0, 0, 0, 0] or dynamic_para) \
          and not switch_to_general_scheme:
        pattc = DeConvKernelSize1Pattern(filter_sizes,
                                         strides=strides,
                                         pad=padding,
                                         output_shape=shape_dx,
                                         fusion_para=fusion_para,
                                         dynamic_para=dynamic_para,
                                         kernel_name=kernel_name)
    else:
        pattc = DeConvPattern(filter_sizes, strides=strides,
                              pad=padding, output_shape=shape_dx,
                              dilations=dilations, offset_x=offset_x,
                              fusion_para=fusion_para,
                              dynamic_para=dynamic_para,
                              kernel_name=kernel_name)

    dy_col = pattc.generate_a(out_backprop)
    w_col = pattc.generate_b(filters)
    dx_ddr = pattc.generate_c(dy_col, w_col, tensor_bias=tensor_bias)

    return dx_ddr


class DynamicConv2dBpInputParams:

    dynamic_mode = None
    tiling_info_dict = {}
