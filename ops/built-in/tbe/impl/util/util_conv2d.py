#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

conv2d common
provide common function used by conv2d
"""

import math
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils.error_manager import error_manager_conv2d as err_man


PAD_SHAPE_DIM = 2


def lcm(a_val, b_val):
    return (a_val*b_val)//math.gcd(a_val, b_val)


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


def calc_para_from_tensor(inputs, weights, bias, offset_w, strides, pads,
                          dilations, offset_x, groups, kernel_name,
                          data_format="NCHW", options=None):

    shape_w = []
    for i in weights.op.attrs['ori_shape']:
        shape_w.append(i.value)
    shape_fm = []
    for i in inputs.shape:
        shape_fm.append(i.value)

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
    shape_c = shape_w[pos_c]*groups
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

    fusion_para = _conv2d_compute_fusion_para(inputs)

    valid_shape = fusion_para.get("valid_shape")
    if valid_shape and valid_shape[2] == shape_fm[2]:
        valid_shape = ()
        fusion_para["valid_shape"] = ()
        fusion_para["slice_offset"] = ()
    if valid_shape:
        input_h = valid_shape[2]
        input_w = valid_shape[3]

    strideh = _trans_stride(input_h, weight_h, strideh, padh, dlt_h)
    stridew = _trans_stride(input_w, weight_w, stridew, padw, dlt_w)

    c0_val = 16
    if weights.dtype == "int8":
        c0_val = 32
    cin_ori = shape_c//groups
    cout_ori = cout_all//groups
    enlarge = min(lcm(lcm(cin_ori, c0_val)//cin_ori, lcm(cout_ori, 16)//cout_ori), groups)
    c1_opt = math.ceil(cin_ori*enlarge/c0_val)
    cout1_opt = math.ceil(cout_ori*enlarge/16)
    group_opt = math.ceil(groups/enlarge)

    para_dict = {"pad_h": padh, "pad_w": padw, "stride_h": strideh,
                 "stride_w": stridew, "dilate_h": dlt_h, "dilate_w": dlt_w,
                 "offset_x": offset_x, "filter_h": weight_h,
                 "filter_w": weight_w, "bias_tensor": bias,
                 "offset_w_tensor": offset_w,
                 "fusion_para": fusion_para,
                 "kernel_name": kernel_name,
                 "group": groups,
                 "enlarge": enlarge,
                 "c1_opt": c1_opt,
                 "cout1_opt": cout1_opt,
                 "group_opt": group_opt,
                 "a_shape": _shape_to_list(inputs.shape),
                 "weight_fracz_shape": _shape_to_list(weights.shape),
                 "weight_ori_shape_nchw": [cout_all, shape_c, weight_h, weight_w]}
    c0_optim_flg = False
    use_v200_c04_flg = False
    if shape_c <= 4 and ("format" in weights.op.attrs and
                         weights.op.attrs['format'].value == "FRACTAL_Z_C04"):
        c0_optim_flg = True
        if (weight_h == 1) and (weight_w == 1):
            err_man.raise_err_specific_user("conv2d", "weight shape does "\
                + "not support that H and W are both equal to 1 when C0=4.")

        if fusion_para["input_memory_type"] == 1:
            err_man.raise_err_specific_input_shape("conv2d", "c0 optim not "\
                + "support fmap from L1 directly (instead of DDR)")
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
        err_man.raise_err_input_format_invalid("conv2d", \
            "inputs", ["NCHW", "NHWC"], format_x)
    pos_n = format_x.find('N')
    pos_c = format_x.find('C')
    pos_h = format_x.find('H')
    pos_w = format_x.find('W')
    shape_fm = [shape_x[pos_n], shape_x[pos_c], shape_x[pos_h], shape_x[pos_w]]

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

    valid_shape = fusion_para.get("valid_shape")
    if valid_shape and valid_shape[2] == shape_fm[2]:
        valid_shape = ()
        fusion_para["valid_shape"] = ()
        fusion_para["slice_offset"] = ()

    if valid_shape:
        input_h = valid_shape[2]
        input_w = valid_shape[3]
    else:
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
        if fusion_para["input_memory_type"] == 1:
            err_man.raise_err_specific_input_shape("conv2d", \
                "c0 optim not support fmap "\
                + "from L1 directly (instead of DDR)")
        if inputs.get("format") == "NC1HWC0_C04":
            use_v200_c04_flg = True

    optim_dict = {"c0_optim_flg": c0_optim_flg,
                  "use_v200_c04_flg": use_v200_c04_flg}

    return shape_fm, shape_filter, padh, padw, strideh, stridew, \
           dlt_h, dlt_w, optim_dict, fusion_para


@para_check.check_input_type((list, tuple), (list, tuple), (list, int), (list, int),
                       int, int, str, str, str, str,
                       bool, str, int, int, dict, dict, int)
def conv_layer_cce_para_check(shape_in, shape_w, padh, padw, strideh, stridew,
                              in_dtype, w_dtype, res_dtype, offset_w_dtype,
                              bias, kernel_name, dilateh=1, dilatew=1,
                              optim_dict=None, fusion_para=None, groups=1):
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

    fusion_para: the config for L1 or L2 Fusion

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

    if fusion_para is None:
        fusion_para = {"input_memory_type": 0, "output_memory_type": 0,
                       "valid_shape": (), "slice_offset": (), \
                       "l1_fusion_type": -1, \
                       "fmap_l1_addr_flag": 0, \
                       "fmap_l1_valid_size": -1}

    shape_in, shape_w = tbe.check_conv_shape(shape_in, shape_w,
                                         pad_top, pad_bottom,
                                         pad_left, pad_right, strideh, stridew,
                                         in_dtype, w_dtype, fusion_para,
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
    block_size_k = tbe_platform.CUBE_MKN[in_dtype]['mac'][1]
    if optim_dict["c0_optim_flg"] and optim_dict["use_v200_c04_flg"] \
            and tbe_platform.get_soc_spec("SOC_VERSION") in \
            ("Ascend710", "Ascend615", "Ascend610", "Hi3796CV300CS"):
        block_size_k = 4
    fmap_shape_nc1hwc0 = (shape_in[0], c1in_ori_align,
                          shape_in[2], shape_in[3], block_size_k)

    out_channel, _, filter_h, filter_w = shape_w
    block_size_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
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


def _conv2d_compute_fusion_para(inputs):
    """
    get L2 fusion para for conv2d_compute
    """
    input_memory_type = inputs.op.attrs["addr_type"].value \
        if "addr_type" in inputs.op.attrs else 0
    valid_shape = inputs.op.attrs["valid_shape"] \
        if "valid_shape" in inputs.op.attrs else ()
    slice_offset = inputs.op.attrs["slice_offset"] \
        if "slice_offset" in inputs.op.attrs else ()
    l1_fusion_type = inputs.op.attrs["L1_fusion_type"].value \
    if "L1_fusion_type" in inputs.op.attrs else -1

    fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"].value \
        if "L1_addr_flag" in inputs.op.attrs else "nothing"
    fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"].value \
        if "L1_valid_size" in inputs.op.attrs else -1

    l1_fusion_enable_flag = tbe_platform.get_L1_info("L1_fusion_enabled")
    if not l1_fusion_enable_flag:
        l1_fusion_type = -1

    valid_shape = _shape_to_list(valid_shape)
    slice_offset = _shape_to_list(slice_offset)

    l2_fusion_enable_flag = tbe_platform.get_L1_info("L2_fusion_enabled")
    l1_fusion_enable_flag = tbe_platform.get_L1_info("L1_fusion_enabled")

    if (not l2_fusion_enable_flag) and (not l1_fusion_enable_flag):
        input_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1

    if (l2_fusion_enable_flag or (not l1_fusion_enable_flag)) and (input_memory_type == 1 or l1_fusion_type != -1):
        err_man.raise_err_specific_user("conv2d", "if enable L2 fusion and"\
            + "not enable L1 fusion, input_memory_type must not be 1 or L1 fusion type can't equal -1")

    if input_memory_type not in (0, 1, 2):
        err_man.raise_err_input_mem_type("conv2d", input_memory_type)
    if valid_shape and not slice_offset:
        err_man.raise_err_specific_user("conv2d", "if valid_shape exists "\
            + "slice_offset can not be []")

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": "fuse_flag",
                   "valid_shape": valid_shape, "slice_offset": slice_offset,
                   "l1_fusion_type": l1_fusion_type, \
                   "fmap_l1_addr_flag": fmap_l1_addr_flag, \
                   "fmap_l1_valid_size": fmap_l1_valid_size}

    return fusion_para


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
                    (kernel - 1)*dlt + 1 else stride


def _conv2d_fusion_para(inputs, outputs):
    """
    get L2 fusion para for conv2d
    """
    input_memory_type = inputs.get("addr_type") \
        if "addr_type" in inputs else 0
    output_memory_type = outputs.get("addr_type") \
        if "addr_type" in outputs else 0
    valid_shape = inputs.get("valid_shape") \
        if "valid_shape" in inputs else ()
    slice_offset = inputs.get("slice_offset") \
        if "slice_offset" in inputs else ()
    l1_fusion_type = inputs.get("L1_fusion_type") \
        if "L1_fusion_type" in inputs else -1

    fmap_l1_addr_flag = inputs.get("L1_addr_flag", "nothing")
    fmap_l1_valid_size = inputs.get("L1_valid_size", -1)

    l1_fusion_enable_flag = tbe_platform.get_L1_info("L1_fusion_enabled")
    if not l1_fusion_enable_flag:
        l1_fusion_type = -1

    valid_shape = _shape_to_list(valid_shape)
    slice_offset = _shape_to_list(slice_offset)

    l2_fusion_enable_flag = tbe_platform.get_L1_info("L2_fusion_enabled")

    if not l2_fusion_enable_flag and (not l1_fusion_enable_flag):
        input_memory_type = 0
        output_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1

    if (l2_fusion_enable_flag or (not l1_fusion_enable_flag)) and (input_memory_type == 1 or l1_fusion_type != -1):
        err_man.raise_err_specific_user("conv2d", "if enable L2 fusion and"\
            + "not enable L1 fusion, input_memory_type must not be 1 or L1 fusion type can't equal -1")

    if input_memory_type not in (0, 1, 2):
        err_man.raise_err_input_mem_type("conv2d", input_memory_type)
    if output_memory_type not in (0, 1, 2):
        err_man.raise_err_output_mem_type("conv2d", output_memory_type)
    if valid_shape and not slice_offset:
        err_man.raise_err_specific_user("conv2d", "if valid_shape exists "\
           + "slice_offset can not be []")

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": output_memory_type,
                   "valid_shape": valid_shape, "slice_offset": slice_offset, \
                   "l1_fusion_type": l1_fusion_type, \
                   "fmap_l1_addr_flag": fmap_l1_addr_flag, \
                   "fmap_l1_valid_size": fmap_l1_valid_size}

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
    filter_h_dilation = (shape_filter[2] - 1)*dilate_h + 1
    filter_w_dilation = (shape_filter[3] - 1)*dilate_w + 1
    w_out = (shape_fm[3] + (pad_left + pad_right) - filter_w_dilation) // stridew + 1
    h_out_part = _lcm(w_out, 16) // w_out
    h_part_length = (h_out_part - 1) * strideh + filter_h_dilation - (pad_top + pad_bottom)
    minimum_load_L1 = 2 * 1 * 4 * h_part_length * shape_fm[3]
    return minimum_load_L1


def use_v200_c04_check(shape_fm, shape_filter, params):
    """
    Check whether to use v200 c0=4 optimization
    """
    use_v200_c04_flg = False
    strides, pads, dilations, data_format = params[5], params[6], params[7], params[9]
    minimum_load_L1 = _get_minimum_load_L1(shape_fm, shape_filter, strides, pads, dilations, data_format)
    if minimum_load_L1 < tbe_platform.get_soc_spec("L1_SIZE"):
        use_v200_c04_flg = True
    return use_v200_c04_flg
