#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

conv2d common
provide common function used by conv2d
"""

import math
from tbe import tvm
from tbe.common.utils import para_check
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform import CUBE_MKN
from tbe.common.utils.errormgr import error_manager_cube as err_man
from topi.cce import util


PAD_SHAPE_DIM = 2
# fmapH, fmapW must be in [1,4096]
FMAP_HW_MIN = 1
FMAP_W_MAX = 2**32-1
FMAP_H_MAX = 100000
DMA_HW_MAX = 2**32-1

FMAP_W_MIN_SPLIT_W = 1
FMAP_W_MAX_SPLIT_W = 4294967295

# filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

# padH, padW must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# stride must be in [1,63]
STRIDE_MIN = 1
STRIDE_MAX = 63

# dilate must be in [1,255]
DILATE_MIN = 1
DILATE_MAX = 255
CONV_SHAPE_DIM = 4

# In v200, small channel case: 4*filter_h*filter_w must be smaller than 65536.
HK_WK_C04_V200 = 65535

def lcm(a_val, b_val):
    return (a_val * b_val) // math.gcd(a_val, b_val)


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    if isinstance(shape, (list, tuple)):
        return shape
    tmp = []
    if shape == "":
        return ()
    for i in shape:
        tmp.append(i.value)
    return tmp

def is_support_v200():
    """
    Check if Ascend610/Ascend615/Ascend710/Hi3796CV300CS version.
    ----------

    Returns
    -------
    True:  Ascend610/Ascend615/Ascend710/Hi3796CV300CS version
    False: Other version
    """
    soc_version = get_soc_spec("SOC_VERSION")
    if soc_version in ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403"):
        return True
    return False


def check_conv_shape(shape_in, shape_w, pad_top, pad_bottom,
                     pad_left, pad_right, strideh, stridew, in_dtype, w_dtype,
                     optim_dict=None, dilateh=1, dilatew=1, dynamic_para=None, groups=1):
    """

    Parameters
    ----------
    shape_in : shape of data_in

    shape_w : shape of filter

    padh: the padding shape in H

    padw: the padding shape in weight

    strideh: the stride value in H

    stridew: the stride value in weight

    dilateh: the dilate value in H

    dilatew: the dilate value in weight

    optim_dict: optimize feature dict

    in_dtype : the feature map data type

    w_dtype : the weight data type

    Returns
    -------
    None

    """
    def conv1d_split_w_flag_set():
        """
        For load2d case and load3d cases, set a conv1d_split_w_flag and
        some checks do not apply to conv1D
        """
        conv1d_split_w_flag = shape_in[2] == 1 and shape_w[2] == 1 and pad_top == 0 and pad_bottom == 0
        return conv1d_split_w_flag

    def load2d_split_w_flag_set():
        """
        Set a flag for load2d to replace the load3d
        and can split w
        Some checks do not apply to conv1D.
        """
        load2d_to_load3d_flag = (shape_w[2] == 1) and (shape_w[3] == 1) and \
                                (pad_top == 0) and (pad_bottom == 0) and \
                                (pad_left == 0) and (pad_right == 0) and \
                                (strideh == 1) and (stridew == 1) and \
                                (in_dtype == "float16") and \
                                (w_dtype == "float16") and (shape_in[2] == 1)
        return load2d_to_load3d_flag

    def dilate_check():
        """
        Check dilate.
        """

        if dilateh < DILATE_MIN or dilateh > DILATE_MAX:
            range_value = "".join([str(DILATE_MIN), ", ", str(DILATE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "dilateh", str(dilateh))
        if dilatew < DILATE_MIN or dilatew > DILATE_MAX:
            range_value = "".join([str(DILATE_MIN), ", ", str(DILATE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "dilatew", str(dilatew))

    conv1d_split_w_flag = conv1d_split_w_flag_set()

    def check_fm_w_flag_set():
        """
        Check fmap split width flag.
        """
        check_fm_w_flag = False
        check_fm_w_flag = (int(shape_in[3]) < FMAP_HW_MIN or int(shape_in[3]) > FMAP_W_MAX) and not conv1d_split_w_flag
        return check_fm_w_flag

    def _check_fmap_range(fmap_range):
        """
        Check fmap range.
        """

        def _check_h_range():
            if int(shape_in[2]) < FMAP_HW_MIN or int(shape_in[2]) > DMA_HW_MAX:
                range_value = "".join([str(FMAP_HW_MIN), ", ", str(DMA_HW_MAX)])
                err_man.raise_err_attr_range_invalid("conv2d", range_value, "feature map H", shape_in[2])

        def _check_w_range():
            if check_fm_w_flag_set():
                range_value = "".join([str(FMAP_HW_MIN), ", ", str(FMAP_W_MAX)])
                err_man.raise_err_attr_range_invalid("conv2d", range_value, "feature map W", shape_in[3])
            if conv1d_split_w_flag and (shape_in[3] < FMAP_W_MIN_SPLIT_W or shape_in[3] > FMAP_W_MAX_SPLIT_W):
                range_value = "".join([str(FMAP_W_MIN_SPLIT_W), ", ", str(FMAP_W_MAX_SPLIT_W)])
                err_man.raise_err_attr_range_invalid("conv2d", range_value, "feature map W when split w", shape_in[3])
        _check_h_range()
        _check_w_range()

    fmap_range = None if dynamic_para is None else dynamic_para.get("fmap_range")
    _check_fmap_range(fmap_range)
    if not dynamic_para:
        util.check_shape_rule(shape_in, CONV_SHAPE_DIM, CONV_SHAPE_DIM)
    util.check_shape_rule(shape_w, CONV_SHAPE_DIM, CONV_SHAPE_DIM)

    if shape_in[1] != shape_w[1]:
        err_man.raise_err_scene_equal_limitation("conv2d", "input feature map channel", "filter channel")

    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False, "use_v200_c04_flg": False}
    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    shape_in[1] = ((shape_in[1] + block_size_k - 1) // block_size_k)*block_size_k
    # int8 feature_map_channel_in is aligned by 16, but weight_channel_in is aligned by 32.
    shape_w[1] = ((shape_in[1] + block_size_k - 1) // block_size_k)*block_size_k
    if optim_dict["c0_optim_flg"]:
        shape_in[1] = 4
        shape_w[1] = 4
    h_i = shape_in[2]
    w_i = shape_in[3]
    h_k = shape_w[2]
    w_k = shape_w[3]

    # dilateh, dilatew check
    dilate_check()

    hk_dilation = (h_k - 1) * dilateh + 1
    wk_dilation = (w_k - 1) * dilatew + 1

    h_out = (h_i + pad_top + pad_bottom - hk_dilation) // strideh + 1
    w_out = (w_i + pad_left + pad_right - wk_dilation) // stridew + 1
    if not dynamic_para and (int(w_out) < 1 or int(h_out) < 1):
        err_man.raise_err_specific("conv2d", "output shape should greater than 0, please check the input shape.\n")

    def _check_pad():
        """
        Check pad.
        """
        # padh, padw check
        if isinstance(pad_top, tvm.expr.Expr) or isinstance(pad_bottom, tvm.expr.Expr) or \
                isinstance(pad_left, tvm.expr.Expr) or isinstance(pad_right, tvm.expr.Expr):
            return
        if pad_top < PAD_MIN or pad_bottom < PAD_MIN or \
                pad_top > PAD_MAX or pad_bottom > PAD_MAX:
            range_value = "".join([str(PAD_MIN), ", ", str(PAD_MAX)])
            actual_value = "".join([str(pad_top), ", ", str(pad_bottom)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "pad_top or pad_bottom", actual_value)
        if pad_left < PAD_MIN or pad_right < PAD_MIN or \
                pad_left > PAD_MAX or pad_right > PAD_MAX:
            range_value = "".join([str(PAD_MIN), ", ", str(PAD_MAX)])
            actual_value = "".join([str(pad_left), ", ", str(pad_right)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "pad_left or pad_right", actual_value)

    w_block_size_n = CUBE_MKN[w_dtype]['mac'][2]
    shape_w[0] = ((shape_w[0] + w_block_size_n - 1) // w_block_size_n) * w_block_size_n

    # filterH, filterW check(before dilation according to chip design demand )
    def _check_w_range():
        """
        Check width shape.
        """
        if shape_w[2] < FILTER_HW_MIN or shape_w[2] > DMA_HW_MAX:
            range_value = "".join([str(FILTER_HW_MIN), ", ", str(DMA_HW_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "kernel H", str(shape_w[2]))
        if shape_w[3] < FILTER_HW_MIN or shape_w[3] > DMA_HW_MAX:
            range_value = "".join([str(FILTER_HW_MIN), ", ", str(DMA_HW_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "kernel W", str(shape_w[3]))
        temp = 4*shape_w[2]*shape_w[3]
        if optim_dict.get("use_v200_c04_flg") and is_support_v200() and (temp > HK_WK_C04_V200):
            err_man.raise_err_specific("conv2d", "In v200, small channel case, the 4*Hk*Wk must be smaller than " +
                                       "or equal to " + str(HK_WK_C04_V200) +
                                       ". you can try to disable the small channel.")

    def _check_stride():
        """
        Check stride.
        """
        if strideh < STRIDE_MIN or strideh > STRIDE_MAX:
            range_value = "".join([str(STRIDE_MIN), ", ", str(STRIDE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "strideh", str(strideh))
        if stridew < STRIDE_MIN or stridew > STRIDE_MAX:
            range_value = "".join([str(STRIDE_MIN), ", ", str(STRIDE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "stridew", str(stridew))
    _check_w_range()
    _check_pad()
    _check_stride()

    config = CUBE_MKN[w_dtype]
    ci0 = config['mac'][1]
    if ci0 <= 0:
        err_man.raise_err_specific("conv2d", "ci0 must > 0")
    shape_in_fusion_para_check = shape_in
    shape_in_fusion_para_check[1] = shape_in_fusion_para_check[1] // groups

    # check for not bigger than L1
    m_bit_ratio = {"float16": 2, "int8": 1}
    point_per_w = math.floor((w_i - wk_dilation + pad_left + pad_right) / stridew) + 1
    w_in = math.floor(config['mac'][0] / point_per_w) + 2
    tmp = ((int(w_in) - 1) * strideh + hk_dilation) * w_i

    return shape_in, shape_w


def calc_para_from_tensor(inputs, weights, bias, offset_w, strides, pads,
                          dilations, offset_x, groups, kernel_name,
                          data_format="NCHW", options=None):

    shape_w = []
    for i in weights.op.attrs['ori_shape']:
        shape_w.append(i.value)
    shape_fm = []
    multi_conv2d_fusion_flag = False
    if len(inputs.shape) == 5:
        for i in inputs.shape:
            shape_fm.append(i.value)
    elif len(inputs.shape) == 4:
        if inputs.op.attrs['current_shape']:
            cur_shape = inputs.op.attrs['current_shape']
            if cur_shape[2].value * cur_shape[3].value != inputs.shape[2].value:
                err_man.raise_err_specific_input_shape("conv2d",
                                                       "the h*w of current_shape is not equal inputs.shape[3].value")
            multi_conv2d_fusion_flag = True
            for i in inputs.op.attrs['current_shape']:
                shape_fm.append(i.value)
        else:
            err_man.raise_err_specific("conv2d", "current_shape not in op.attrs on 4 dimensions tensor")
    else:
        err_man.raise_err_input_params_not_expected("conv2d", "fmap", "4 dimensions or 5 dimensions",
                                                    str(len(inputs.shape)) + " dimensions")

    input_h = shape_fm[2]
    input_w = shape_fm[3]

    format_w = weights.op.attrs['ori_format'].value
    all_fmt = ["NCHW", "NHWC", "HWCN"]
    if format_w not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", \
        "weights", ["NCHW", "NHWC", "HWCN"], format_w)

    pos_c = format_w.find('C')
    pos_h = format_w.find('H')
    pos_w = format_w.find('W')
    pos_cout = format_w.find('N')
    weight_h = shape_w[pos_h]
    weight_w = shape_w[pos_w]
    # fix the weight's channel=cin_ori
    shape_c = shape_w[pos_c] * groups
    cout_all = shape_w[pos_cout]

    if len(strides) != 4:
        err_man.raise_err_should_be_4d("conv2d", "strides")
    if len(dilations) != 4:
        err_man.raise_err_should_be_4d("conv2d", "directions")

    pos_h = data_format.find('H')
    pos_w = data_format.find('W')
    strideh = strides[pos_h]
    stridew = strides[pos_w]
    dlt_h = dilations[pos_h]
    dlt_w = dilations[pos_w]

    if len(pads) == 4:
        padh = [pads[0], pads[1]]
        padw = [pads[2], pads[3]]
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")

    strideh = _trans_stride(input_h, weight_h, strideh, padh, dlt_h)
    stridew = _trans_stride(input_w, weight_w, stridew, padw, dlt_w)

    c0_val = 16
    if weights.dtype == "int8":
        c0_val = 32
    cin_ori = shape_c // groups
    cout_ori = cout_all // groups
    enlarge = min(lcm(lcm(cin_ori, c0_val) // cin_ori, lcm(cout_ori, 16) // cout_ori), groups)
    c1_opt = math.ceil(cin_ori*enlarge / c0_val)
    cout1_opt = math.ceil(cout_ori*enlarge / 16)
    group_opt = math.ceil(groups / enlarge)

    if inputs.op.tag == "aipp_res_convolution":
        fmap_l1_addr_flag = "nothing"
        fmap_l1_valid_size = -1
        slice_offset = (0, 0, 0, 0, 0)
        from te.tvm.buffer_manager import get_buffer_manager
        buffer_manager = get_buffer_manager()
        for remapped_buffer in buffer_manager.get_remapped_buffers():
            remapped_buffer_attr = remapped_buffer.get_buffer_attr()
            if "L1_addr_flag" in remapped_buffer_attr and remapped_buffer_attr["L1_addr_flag"] != "nothing":
                fmap_l1_addr_flag = remapped_buffer_attr.get("L1_addr_flag", "nothing")
                fmap_l1_valid_size = remapped_buffer_attr.get("L1_valid_size", -1)
                slice_offset = remapped_buffer_attr.get("slice_offset", (0, 0, 0, 0, 0))
                break
    else:
        fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"].value if "L1_addr_flag" in inputs.op.attrs else "nothing"
        fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"].value if "L1_valid_size" in inputs.op.attrs else -1
        slice_offset = inputs.op.attrs["slice_offset"] if "slice_offset" in inputs.op.attrs else (0, 0, 0, 0, 0)
    fusion_para = {"fmap_l1_addr_flag": fmap_l1_addr_flag,
                   "fmap_l1_valid_size": fmap_l1_valid_size,
                   "slice_offset": slice_offset}

    para_dict = {"pad_h": padh,
                 "pad_w": padw,
                 "stride_h": strideh,
                 "stride_w": stridew,
                 "dilate_h": dlt_h,
                 "dilate_w": dlt_w,
                 "offset_x": offset_x,
                 "filter_h": weight_h,
                 "filter_w": weight_w,
                 "bias_tensor": bias,
                 "offset_w_tensor": offset_w,
                 "fusion_para": fusion_para,
                 "kernel_name": kernel_name,
                 "group": groups,
                 "enlarge": enlarge,
                 "c1_opt": c1_opt,
                 "cout1_opt": cout1_opt,
                 "group_opt": group_opt,
                 "a_shape": shape_fm,
                 "weight_fracz_shape": _shape_to_list(weights.shape),
                 "weight_ori_shape_nchw": [cout_all, shape_c, weight_h, weight_w],
                 "multi_conv2d_fusion_flag": multi_conv2d_fusion_flag}
    c0_optim_flg = False
    use_v200_c04_flg = False
    if shape_c <= 4 and ("format" in weights.op.attrs and
                         weights.op.attrs['format'].value == "FRACTAL_Z_C04"):
        c0_optim_flg = True
        if (weight_h == 1) and (weight_w == 1):
            err_man.raise_err_specific_user("conv2d", "weight shape does "\
                + "not support that H and W are both equal to 1 when C0=4.")

        if inputs.shape[-1].value == 4:
            use_v200_c04_flg = True

    optim_dict = {"c0_optim_flg": c0_optim_flg,
                  "use_v200_c04_flg": use_v200_c04_flg,
                  "invalid_data_rm": False}

    if options is not None:
        optim_dict.update(options)

    return para_dict, optim_dict


def calc_para_from_dict(inputs, weights, strides, pads,
                        dilations, outputs, data_format="NCHW"):
    shape_x = inputs.get("ori_shape")
    shape_x_5hd = inputs.get("shape")
    shape_w = weights.get("ori_shape")

    if len(strides) != 4:
        err_man.raise_err_should_be_4d("conv2d", "strides")
    if len(dilations) != 4:
        err_man.raise_err_should_be_4d("conv2d", "dilations")

    if len(pads) == 4:
        padh = [pads[0], pads[1]]
        padw = [pads[2], pads[3]]
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")

    if (not isinstance(shape_x, (tuple, list))) or len(shape_x) != 4:
        err_man.raise_err_should_be_4d("conv2d", "inputs")

    if (not isinstance(shape_w, (tuple, list))) or len(shape_w) != 4:
        err_man.raise_err_should_be_4d("conv2d", "weights")

    format_x = inputs.get("ori_format")
    all_fmt = ["NCHW", "NHWC"]
    if format_x not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", "inputs", ["NCHW", "NHWC"], format_x)

    pos_c = format_x.find('C')
    # only c is original value when lxfusion split batch and h.
    shape_fm = [shape_x_5hd[0], shape_x[pos_c], shape_x_5hd[2], shape_x_5hd[3]] # [Ni, Ci, Hi, Wi]

    pos_attr_h = data_format.find('H')
    pos_attr_w = data_format.find('W')
    strideh = strides[pos_attr_h]
    stridew = strides[pos_attr_w]
    dlt_h = dilations[pos_attr_h]
    dlt_w = dilations[pos_attr_w]

    format_w = weights.get("ori_format")
    all_fmt = ["NCHW", "NHWC", "HWCN"]
    if format_w not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", \
            "weights", ["NCHW", "NHWC", "HWCN"], format_w)
    pos_n = format_w.find('N')
    pos_c = format_w.find('C')
    pos_h = format_w.find('H')
    pos_w = format_w.find('W')
    # fix the weight's channel=cin_ori
    shape_filter = [shape_w[pos_n], shape_fm[1], \
                    shape_w[pos_h], shape_w[pos_w]]

    fusion_para = _conv2d_fusion_para(inputs, outputs)

    input_h = shape_fm[2]
    input_w = shape_fm[3]

    strideh = _trans_stride(input_h, shape_filter[2], strideh, padh, dlt_h)
    stridew = _trans_stride(input_w, shape_filter[3], stridew, padw, dlt_w)

    c0_optim_flg = False
    use_v200_c04_flg = False
    if shape_w[pos_c] <= 4 and weights.get("format") == "FRACTAL_Z_C04":
        c0_optim_flg = True
        if (shape_w[pos_h] == 1) and (shape_w[pos_w] == 1):
            err_man.raise_err_specific_user("conv2d", "weight shape "\
                + "does not support that H and W are both "\
                + "equal to 1 when C0=4.")
        if inputs.get("format") == "NC1HWC0_C04":
            use_v200_c04_flg = True

    optim_dict = {"c0_optim_flg": c0_optim_flg,
                  "use_v200_c04_flg": use_v200_c04_flg}

    return shape_fm, shape_filter, padh, padw, strideh, stridew, \
           dlt_h, dlt_w, optim_dict, fusion_para


@para_check.check_input_type((list, tuple), (list, tuple), (list, int), (list, int),
                             int, int, str, str, str, str,
                             bool, str, int, int, dict, int)
def conv_layer_cce_para_check(shape_in, shape_w, padh, padw, strideh, stridew,
                              in_dtype, w_dtype, res_dtype, offset_w_dtype,
                              bias, kernel_name, dilateh=1, dilatew=1,
                              optim_dict=None, groups=1):
    """

    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    padh: H direction padding

    padw: W direction padding

    strideh: H direction stride

    stridew: W direction stride

    in_dtype: the feature map data type

    w_dtype: the weight data type

    res_dtype: the result data type

    offset_w_dtype: weight offset data type, default 'int32'

    bias: the tag for bias or not

    kernel_name: cce kernel name

    dilateh: H direction spacing between kernel

    dilatew: W direction spacing between kernel

    optim_dict: optimize feature dict

    Returns
    -------
    None

    """
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(offset_w_dtype, ['int32'])
    para_check.check_dtype_rule(in_dtype, ('int8', "float16"))
    para_check.check_dtype_rule(w_dtype, ('int8', "float16"))
    para_check.check_dtype_rule(res_dtype, ('int32', "float16"))

    if isinstance(padh, list):
        if len(padh) != PAD_SHAPE_DIM:
            err_man.raise_err_specific_user("conv2d", "Dimension must be "\
                                            + str(PAD_SHAPE_DIM) + \
                                            " when padh is a list.")
        pad_top = padh[0]
        pad_bottom = padh[1]
    else:
        pad_top = padh
        pad_bottom = padh

    if isinstance(padw, list):
        if len(padw) != PAD_SHAPE_DIM:
            err_man.raise_err_specific_user("conv2d", "Dimension must be "\
                                            + str(PAD_SHAPE_DIM) + \
                                            " when padw is a list.")
        pad_left = padw[0]
        pad_right = padw[1]
    else:
        pad_left = padw
        pad_right = padw
    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False, "use_v200_c04_flg": False}
    optim_off = shape_in[1] > 4 or shape_w[1] > 4 or \
                (shape_w[2] == 1 and shape_w[3] == 1)
    if optim_dict.get("c0_optim_flg") is True:
        if optim_off:
            err_man.raise_err_specific_user("conv2d", "Invalid "\
                + "config for c0=4 optimize feature.")

    shape_in, shape_w = check_conv_shape(shape_in, shape_w,
                                         pad_top, pad_bottom,
                                         pad_left, pad_right, strideh, stridew,
                                         in_dtype, w_dtype,
                                         optim_dict, dilateh, dilatew, groups=groups)

    return shape_in, shape_w


def conv_layer_cce_shape_calc(shape_in, shape_w, in_dtype, \
    w_dtype, optim_dict, cout1_opt=1, c1_opt=1, group_opt=1, c1in_ori_align=1):
    """

    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    in_dtype: the feature map data type

    w_dtype: the weight data type

    optim_dict: optimize feature dict

    Returns
    -------
    None

    """
    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    if optim_dict["c0_optim_flg"] and optim_dict["use_v200_c04_flg"] \
            and get_soc_spec("SOC_VERSION") in \
            ("Ascend710", "Ascend615", "Ascend610", "Hi3796CV300CS", "SD3403"):
        block_size_k = 4
    fmap_shape_nc1hwc0 = (shape_in[0], c1in_ori_align,
                          shape_in[2], shape_in[3], block_size_k)

    out_channel, _, filter_h, filter_w = shape_w
    block_size_k = CUBE_MKN[w_dtype]['mac'][1]
    block_size_n = CUBE_MKN[w_dtype]['mac'][2]
    if optim_dict["c0_optim_flg"]:
        filter_shape_frac_z = ((4 * filter_h * filter_w + block_size_k - 1) \
                               // block_size_k,
                               out_channel // block_size_n, block_size_n,
                               block_size_k)
    else:
        filter_shape_frac_z = (group_opt * c1_opt * filter_h * filter_w,
                               cout1_opt, block_size_n,
                               block_size_k)
    return fmap_shape_nc1hwc0, filter_shape_frac_z


def _trans_stride(input_size, kernel, stride, pad, dlt):
    """
    transform stride

    Notice
    ------
    adapt stride value to hardware request

    Parameters
    ----------
    input_size: int
        feature map H/W size
    kernel: int
        kernel H/W size
    pad: 2D list of int
        pad on H/W side
    strides: int
        stride on H/W
    dlt: int
        dilation on H/W
    Returns
    -------
    new stride
    """
    return 1 if input_size + pad[0] + pad[1] == \
                    (kernel - 1) * dlt + 1 else stride


def _conv2d_fusion_para(inputs, outputs):
    """
    get lxfusion params for conv2d
    """
    fmap_l1_addr_flag = inputs.get("L1_addr_flag", "nothing")
    fmap_l1_valid_size = inputs.get("L1_valid_size", -1)
    slice_offset = inputs.get("slice_offset", (0, 0, 0, 0, 0))
    fusion_para = {"fmap_l1_addr_flag": fmap_l1_addr_flag,
                   "fmap_l1_valid_size": fmap_l1_valid_size,
                   "slice_offset": slice_offset}

    return fusion_para


def _lcm(num1, num2):
    """
    Obtain the least common multiple of num1 and num2
    """
    tmp = num1 * num2
    while num1 % num2 != 0:
        num1, num2 = num2, (num1 % num2)
    return tmp // num2


def _get_minimum_load_L1(shape_fm, shape_filter, strides, pads, dilations, data_format="NCHW"):
    """
    Obtains the minimum amount of data to be loaded to L1.
    """
    pos_attr_h = data_format.find('H')
    pos_attr_w = data_format.find('W')
    strideh = strides[pos_attr_h]
    stridew = strides[pos_attr_w]
    dilate_h = dilations[pos_attr_h]
    dilate_w = dilations[pos_attr_w]
    if len(pads) == 4:
        pad_top, pad_bottom, pad_left, pad_right = pads
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")
    strideh = _trans_stride(shape_fm[2], shape_filter[2], strideh, [pad_top, pad_bottom], dilate_h)
    stridew = _trans_stride(shape_fm[3], shape_filter[3], stridew, [pad_left, pad_right], dilate_w)
    filter_h_dilation = (shape_filter[2] - 1) * dilate_h + 1
    filter_w_dilation = (shape_filter[3] - 1) * dilate_w + 1
    w_out = (shape_fm[3] + (pad_left + pad_right) - filter_w_dilation) // stridew + 1
    h_out_part = _lcm(w_out, 16) // w_out
    h_part_length = (h_out_part - 1) * strideh + filter_h_dilation
    minimum_load_L1 = 2 * 1 * 4 * h_part_length * shape_fm[3]
    return minimum_load_L1


def use_v200_c04_check(shape_fm, shape_filter, params):
    """
    Check whether to use v200 c0=4 optimization
    """
    use_v200_c04_flg = False
    strides, pads, dilations, data_format = params[5], params[6], params[7], params[9]
    minimum_load_L1 = _get_minimum_load_L1(shape_fm, shape_filter, strides, pads, dilations, data_format)
    if minimum_load_L1 < get_soc_spec("L1_SIZE"):
        use_v200_c04_flg = True
    return use_v200_c04_flg
