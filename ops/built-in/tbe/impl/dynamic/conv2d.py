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
dynamic conv2d
"""
from __future__ import absolute_import

import math
from te import tvm
import te.lang.cce as tbe
import te.lang.dynamic as dynamic
import te.platform as tbe_platform
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils.error_manager import error_manager_conv2d as err_man
from te.utils.error_manager import error_manager_util
from impl.util import fusion_util
from impl.util import util_conv2d

NONETYPE = type(None)
# n, h, w dim in NCHW/NC1HWC0 format
N_DIM = 0
H_DIM = 2
W_DIM = 3
# dim_size of NCHW/NHWC format
FORMAT_4D_DIMS = 4
# dim_size of NC1HWC0 format
FORMAT_5D_DIMS = 5
HW_MIN = 1
HW_MAX = 4096


def _ceil(x_1, x_2):
    if x_2 == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))
    return (x_1 + x_2 - 1) // x_2


def _pos_from_format(ele_format):
    """
    get value from ele_format
    """

    pos_n = ele_format.find('N')
    pos_c = ele_format.find('C')
    pos_h = ele_format.find('H')
    pos_w = ele_format.find('W')
    return pos_n, pos_c, pos_h, pos_w


def _set_default_para():
    """
    set default parameter value
    """

    optim_dict = {"c0_optim_flg": False}
    fusion_para = {"input_memory_type": 0, "output_memory_type": 0,
                   "valid_shape": (), "slice_offset": (),
                   "l1_fusion_type": -1}
    return optim_dict, fusion_para


def _check_dynamic_mode(in_shape, w_shape):
    """
    config dynamic mode
    """

    dynamic_mode = None
    if in_shape[H_DIM] == -1 and in_shape[W_DIM] == -1 \
            and in_shape[N_DIM] != -1 and in_shape[1] != -1 \
            and -1 not in w_shape:
        dynamic_mode = "dynamic_hw"
    elif in_shape[N_DIM] == -1 and in_shape[1] != -1 and in_shape[H_DIM] != -1 \
            and in_shape[W_DIM] != -1 and -1 not in w_shape:
        dynamic_mode = "dynamic_batch"

    if dynamic_mode is None:
        err_man.raise_err_specific_user(
            "conv2d", "dynamic_only support dynamic_hw or dynamic_batch")


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


def _check_4d_len(seq, seq_name):
    if len(seq) != 4:
        err_man.raise_err_should_be_4d("conv2d", seq_name)


def _check_format(param_format, expect_formats, param_name):
    if param_format not in expect_formats:
        err_man.raise_err_input_format_invalid(
            "conv2d", param_name, expect_formats, param_format)


def _check_and_config_para(inputs, weights, bias, offset_w, outputs, strides,
                           pads, dilations, data_format, offset_x, kernel_name):
    """
    check and config dynamic mode
    """

    soc_version = tbe_platform.get_soc_spec("SOC_VERSION")
    if soc_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
        err_man.raise_err_specific_user("conv2d",
            "Hi3796CV300ES and Hi3796CV300CS don't support dynamic shape")

    in_shape = list(inputs.get("ori_shape"))
    w_shape = list(weights.get("ori_shape"))
    in_dtype = inputs.get("dtype")
    w_dtype = weights.get("dtype")
    in_format = inputs.get("ori_format")
    w_format = weights.get("ori_format")
    in_range = inputs.get("range")

    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(in_dtype, ['float16'])
    para_check.check_dtype_rule(w_dtype, ['float16'])
    para_check.check_dtype_rule(outputs.get("dtype"), ['float16'])

    _check_4d_len(in_shape, "in_shape")
    _check_4d_len(w_shape, "weights")
    _check_4d_len(strides, "strides")
    _check_4d_len(dilations, "dilations")
    _check_4d_len(pads, "pads")

    _check_format(data_format, ("NCHW", "NHWC"), "input")
    _check_format(w_format, ("NCHW", "NHWC", "HWCN"), "weights")
    if in_format != data_format:
        err_man.raise_err_specific_user("conv2d", "in_format != data_format")

    in_shape = _get_shape_nchw(in_shape, in_format)
    w_shape = _get_shape_nchw(w_shape, w_format)
    pads, strides, dilations = _get_attrs(pads, strides, dilations, data_format)

    optim_dict, fusion_para = _set_default_para()
    in_shape, w_shape = _round_channel(in_shape, w_shape, in_dtype, w_dtype)
    fmap_range = _get_fmap_range(in_range, in_shape, in_format)
    y_range = _get_y_range(fmap_range, w_shape, pads, strides, dilations)
    batch_range, _, h_range, w_range = fmap_range
    _check_dynamic_mode(in_shape, w_shape)
    _check_variable_range(h_range[0], HW_MIN, HW_MAX, "fmap_h")
    _check_variable_range(h_range[1], HW_MIN, HW_MAX, "fmap_h")
    _check_variable_range(w_range[0], HW_MIN, HW_MAX, "fmap_w")
    _check_variable_range(w_range[1], HW_MIN, HW_MAX, "fmap_w")

    return in_shape, w_shape, pads, strides, dilations, in_dtype, w_dtype, \
        optim_dict, fusion_para, fmap_range, y_range


def _get_output(x_in, k_size, pads, stride, dilation):
    return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1


def _get_y_range(fmap_range, kernel, pads, stride, dilation):
    if -1 in pads:
        y_h_lower = _ceil(fmap_range[H_DIM][0], stride[H_DIM])
        y_h_upper = _ceil(fmap_range[H_DIM][1], stride[H_DIM])
        y_w_lower = _ceil(fmap_range[W_DIM][0], stride[W_DIM])
        y_w_upper = _ceil(fmap_range[W_DIM][1], stride[W_DIM])
    else:
        y_h_lower = _get_output(fmap_range[H_DIM][0], kernel[H_DIM],
                                (pads[0], pads[1]), stride[H_DIM], dilation[H_DIM])
        y_h_upper = _get_output(fmap_range[H_DIM][1], kernel[H_DIM],
                                (pads[0], pads[1]), stride[H_DIM], dilation[H_DIM])
        y_w_lower = _get_output(fmap_range[W_DIM][0], kernel[W_DIM],
                                (pads[2], pads[3]), stride[W_DIM], dilation[W_DIM])
        y_w_upper = _get_output(fmap_range[W_DIM][1], kernel[W_DIM],
                                (pads[2], pads[3]), stride[W_DIM], dilation[W_DIM])
    return [fmap_range[0], (kernel[0], kernel[0]),
            (y_h_lower, y_h_upper), (y_w_lower, y_w_upper)]


def _round_channel(shape_in, shape_w, in_dtype, w_dtype):
    if shape_in[1] != shape_w[1]:
        err_man.raise_err_scene_equal_limitation(
            "conv2d", "input feature map channel", "filter channel")

    block_size_k = tbe_platform.CUBE_MKN[in_dtype]['mac'][1]
    shape_in[1] = _ceil(shape_in[1], block_size_k) * block_size_k
    shape_in[1] = ((shape_in[1] + block_size_k - 1) //
                   block_size_k) * block_size_k

    shape_w[1] = _ceil(shape_in[1], block_size_k) * block_size_k
    w_block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    shape_w[0] = _ceil(shape_w[0], w_block_size_n) * w_block_size_n

    return shape_in, shape_w


def _get_fmap_range(in_range, in_shape, in_format):
    if len(in_range) == FORMAT_4D_DIMS:
        pos_n, pos_c, pos_h, pos_w = _pos_from_format(in_format)
        fmap_range = [in_range[pos_n], in_range[pos_c],
                      in_range[pos_h], in_range[pos_w]]
    # range in NC1HWC0 format sometimes
    elif len(in_range) == FORMAT_5D_DIMS:
        fmap_range = [in_range[N_DIM], (in_shape[1], in_shape[1]),
                      in_range[H_DIM], in_range[W_DIM]]
    else:
        raise RuntimeError("range format should be same as input format")
    return [tuple(r) for r in fmap_range]


def _get_attrs(padding, strides, dilations, data_format):
    pos_n, pos_c, pos_h, pos_w = _pos_from_format(data_format)
    dilations = [dilations[pos_n], dilations[pos_c], dilations[pos_h], dilations[pos_w]]
    strides = [strides[pos_n], strides[pos_c], strides[pos_h], strides[pos_w]]

    return padding, strides, dilations


def _get_shape_nchw(shape_in, format_in):
    pos_n, pos_c, pos_h, pos_w = _pos_from_format(format_in)
    return [shape_in[pos_n], shape_in[pos_c], shape_in[pos_h], shape_in[pos_w]]


def _calc_shape(shape_in, shape_w, in_dtype, w_dtype, optim_dict):
    """
    calculate shape
    """

    batch_size, in_channel, feature_map_h, feature_map_w = shape_in
    block_size_k = tbe_platform.CUBE_MKN[in_dtype]['mac'][1]
    fmap_shape_nc1hwc0 = [batch_size,
                          (in_channel + block_size_k - 1) // block_size_k,
                          feature_map_h, feature_map_w, block_size_k]

    out_channel, in_channel_weight, filter_h, filter_w = shape_w
    block_size_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    if optim_dict["c0_optim_flg"]:
        filter_shape_frac_z = (
            (4 * filter_h * filter_w + block_size_k - 1) // block_size_k,
            out_channel // block_size_n, block_size_n, block_size_k)
    else:
        filter_shape_frac_z = (
            in_channel_weight * filter_h * filter_w // block_size_k,
            out_channel // block_size_n, block_size_n, block_size_k)
    return fmap_shape_nc1hwc0, filter_shape_frac_z


def _calc_pads(fmap_shape_nc1hwc0, shape_filter, strides, dilations, pads):
    """
    calculate pads
    """
    if -1 in pads:
        fmap_h, fmap_w = fmap_shape_nc1hwc0[H_DIM], fmap_shape_nc1hwc0[W_DIM]
        _, _, filter_h, filter_w = shape_filter
        _, _, stride_h, stride_w = strides
        _, _, dilation_h, dilation_w = dilations

        filter_h_dilation = (filter_h - 1) * dilation_h + 1
        filter_w_dilation = (filter_w - 1) * dilation_w + 1
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
        return pad_up, pad_down, pad_left, pad_right
    else:
        return pads

@tbe_base.register_fusion_compute("Conv2D")
def conv2d_fusion_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                          groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                          dsl_flag=True):
    fusion_util.check_fusion_input([inputs])
    fusion_util.check_fusion_input([weights])

    # set fusion build config
    build_cfg = tbe_platform.get_fusion_build_cfg()
    build_cfg['constant_realize_extent_in_infer_bound'] = False

    return _conv2d_compute(
        inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
        groups, data_format, offset_x, kernel_name, dsl_flag)


def _conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                    groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                    dsl_flag=True):

    """
    conv2d compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset for fmap

    Returns
    -------
    tvm compute
    """

    if bias:
        raise RuntimeError("bias is not supported yet in dynamic conv2d")
    if offset_w:
        raise RuntimeError("offset_w is not supported yet in dynamic conv2d")
    if groups != 1:
        raise RuntimeError("dynamic conv2d only supports groups=1")
    bias = None
    offset_w = None
    res_dtype = "float16"
    if not (isinstance(inputs, dict) and isinstance(weights, dict)):
        err_man.raise_err_specific_user(
            "conv2d", "In op[inputs], [weights] must be dict")

    shape_fm, shape_filter, pads, strides, dilations, in_dtype, w_dtype, \
        optim_dict, fusion_para, fmap_range, y_range = _check_and_config_para(
            inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
            data_format, offset_x, kernel_name)

    block_size_k = tbe_platform.CUBE_MKN[in_dtype]['mac'][1]
    block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    cin_ori = shape_fm[1] // groups
    cout_ori = shape_filter[0] // groups
    cin_lcm = util_conv2d.lcm(cin_ori, block_size_k) // cin_ori
    cout_lcm = util_conv2d.lcm(cout_ori, block_size_n) // cout_ori
    enlarge = min(util_conv2d.lcm(cin_lcm, cout_lcm), groups)
    c1_opt = math.ceil(cin_ori*enlarge/block_size_k)
    cout1_opt = math.ceil(cout_ori*enlarge/block_size_n)
    group_opt = math.ceil(groups/enlarge)

    fmap_shape_nc1hwc0, filter_shape_frac_z = _calc_shape(
        shape_fm, shape_filter, in_dtype, w_dtype, optim_dict)
    if fmap_shape_nc1hwc0[H_DIM] == -1 and fmap_shape_nc1hwc0[W_DIM] == -1:
        fmap_shape_nc1hwc0[H_DIM] = tbe_base.var("fmap_h", fmap_range[H_DIM])
        fmap_shape_nc1hwc0[W_DIM] = tbe_base.var("fmap_w", fmap_range[W_DIM])
        h_o = tbe_base.var("ho", y_range[H_DIM])
        w_o = tbe_base.var("wo", y_range[H_DIM])
        tbe_base.add_exclude_bound_var(fmap_shape_nc1hwc0[H_DIM])
        tbe_base.add_exclude_bound_var(fmap_shape_nc1hwc0[W_DIM])
        tbe_base.add_exclude_bound_var(h_o)
        tbe_base.add_exclude_bound_var(w_o)
    elif fmap_shape_nc1hwc0[N_DIM] == -1:
        fmap_shape_nc1hwc0[N_DIM] = tbe_base.var("batch_n", fmap_range[N_DIM])
        tbe_base.add_exclude_bound_var(fmap_shape_nc1hwc0[N_DIM])

    fmap = tvm.placeholder(fmap_shape_nc1hwc0, name='Fmap', dtype=in_dtype)
    weight = tvm.placeholder(filter_shape_frac_z, name='Filter', dtype=w_dtype)

    pad_t, pad_b, pad_l, pad_r = _calc_pads(fmap_shape_nc1hwc0, shape_filter, strides, dilations, pads)
    op_res = tbe.conv(fmap, weight,
                  {"bias_tensor": bias,
                   "offset_w_tensor": offset_w,
                   "pad_h": [pad_t, pad_b], "pad_w": [pad_l, pad_r],
                   "stride_h": strides[H_DIM], "stride_w": strides[W_DIM],
                   "dilate_h": dilations[H_DIM], "dilate_w": dilations[W_DIM],
                   "filter_h": shape_filter[H_DIM],
                   "filter_w": shape_filter[W_DIM],
                   "offset_x": offset_x,
                   "res_dtype": res_dtype,
                   "fusion_para": fusion_para,
                   "kernel_name": kernel_name,
                   "group": groups,
                   "enlarge": enlarge,
                   "c1_opt": c1_opt,
                   "cout1_opt": cout1_opt,
                   "group_opt": group_opt,
                   "a_shape": fmap_shape_nc1hwc0,
                   "weight_fracz_shape": filter_shape_frac_z,
                   "weight_ori_shape_nchw": shape_filter},
                  optim_dict=optim_dict,
                  dsl_flag=dsl_flag)

    return {"op_placeholder": [fmap, weight], "op_res": [op_res]}


@tbe_base.register_operator("Conv2D")
@para_check.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list),
                             int, str, int, str, str)
def conv2d(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
           groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d"):
    """
    algorithm: conv2d

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype and range)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset of fmap
    kernel_name: str
        kernel name, default value is "conv2d"

    Returns
    -------
    None
    """

    with tbe_base.compute():
        res = _conv2d_compute(
            inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
            groups, data_format, offset_x, kernel_name, dsl_flag=False)

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
