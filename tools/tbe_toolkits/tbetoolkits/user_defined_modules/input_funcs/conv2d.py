#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Conv2d input tensor generator"""
# Third-Party Packages
import tbetoolkits
import numpy as np
from .registry import register_input
from ...utilities import get
from ...utilities import get_global_storage


def _ceil(a, b):
    return (a + b - 1) // b


def _align(a, b):
    return _ceil(a, b) * b


def _lcm(param1, param2):
    """
    """
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, (param1 % param2)

    return temp // param2


def _calculate_group(cin, cout, groups):
    block_size = 16

    mag_factor0 = _lcm(cin // groups, block_size) // (cin // groups)
    mag_factor1 = _lcm(cout // groups, block_size) // (cout // groups)
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin_g = _align(mag_factor * (cin // groups), block_size)
    cout_g = _align(mag_factor * (cout // groups), block_size)

    group_dict = {"real_g": _ceil(groups, mag_factor),
                  "mag_factor": mag_factor,
                  "cin_g": cin_g,
                  "cin1_g": cin_g // block_size,
                  "cout_g": cout_g,
                  "cout1_g": cout_g // block_size,
                  "groups": groups,
                  "cin_ori": cin // groups,
                  "cout_ori": cout // groups}
    print('cin:%d, cout:%d, groups:%d, group_dict:' % (cin, cout, groups), group_dict)
    return group_dict


@register_input(["conv2d",
                 "conv2d_relu",
                 "conv2d_leaky_relu",
                 "leaky_relun0_conv2d",
                 "conv2d_add",
                 "conv2d_add_relu",
                 "conv2d_leaky_relu_add",
                 "conv2d_mul",
                 "conv2d_relu6",
                 "conv2d_sigmoid_mul",
                 "conv2d_sigmoid",
                 "conv2d_softplus"])
def _conv2d_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    ipt_0, ipt_1, ipt_2, ipt_3 = context.input_arrays[:4]
    if get(context.dyn_input_formats, 1) == "NC1HWC0":
        input_data_ranges = context.actual_input_data_ranges
        low, high = input_data_ranges[1]
        shape = list(ipt_1.shape)
        N, C1, H, W, C0 = shape
        N = (N + 15) // 16 * 16
        shape = [C1, H, W, N, C0]
        ipt_1 = np.random.uniform(low, high, shape).astype(ipt_1.dtype)
    return (ipt_0, ipt_1, ipt_2, ipt_3), (ipt_0, ipt_1, ipt_2, ipt_3)


@register_input(["depthwise_conv2d"])
def _depthwise_conv2d_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    ipt_0, ipt_1, ipt_2, ipt_3 = context.input_arrays[:4]
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[1]
    shape = list(ipt_1.shape)

    # Cmulti=1
    C_ori, Cmulti, H, W, C0 = shape
    Co1 = (C_ori + 15) // C0
    C = Co1 * C0

    N0 = 16
    Cmulti_1 = (Cmulti + 15) // N0

    ipt_1 = np.zeros([C, Cmulti_1, H, W, N0]).astype(ipt_1.dtype)

    # C,N1,H,W,N0 transto Fracz
    # C, Cmulti_1, H, W, N0 -> C,H,W,Cmulti_1,N0 ->Co1, C0, H , W, Cmulti_1,N0-> C//C0*H*W, Cmulti_1, N0, C0
    ipt_1 = ipt_1.transpose(0, 2, 3, 1, 4).reshape(Co1, C0, H, W, Cmulti_1, N0).transpose(0, 2, 3, 4, 5, 1)

    ipt_1 = ipt_1.reshape(Co1 * H * W, Cmulti_1, 16, 16)
    for i in range(Co1 * H * W):
        for j in range(Cmulti_1):
            ipt_1_ori = np.random.uniform(low, high, [16, 16]).astype(ipt_1.dtype)
            ipt_1_ori = np.multiply(ipt_1_ori, np.eye(16)).astype(ipt_1.dtype)
            ipt_1[i][j] = ipt_1_ori

    return (ipt_0, ipt_1, ipt_2, ipt_3), (ipt_0, ipt_1, ipt_2, ipt_3)


def _conv2d_bp_common(filter_ori, dy_ori, group_dict, block_size=16):
    multi, cin_ori, hk, wk = filter_ori.shape
    batch, cout_ori, ho, wo = dy_ori.shape
    cout1 = _ceil(cout_ori, block_size)
    cout = cout1 * block_size

    real_g = group_dict.get("real_g")
    cin1_g = group_dict.get("cin1_g")
    cin_g = group_dict.get("cin_g")
    cout_g = group_dict.get("cout_g")

    # filter: K, Ci_ori, Hk, Wk -> Co_g, Ci, Hk, Wk -> Ci1, Hk, Wk, Co_g, Ci0
    filter = np.zeros([cout_g, real_g * cin_g, hk, wk], dtype=filter_ori.dtype)
    for n in range(cout_g):
        for c in range(cin_ori):
            if n // multi == c % block_size:
                filter[n, c, :, :] += filter_ori[n % multi, c, :, :]

    filter = filter.reshape(cout_g, real_g * cin1_g, block_size,
                            hk, wk).transpose(1, 3, 4, 0, 2).astype(filter.dtype)

    # dy: N, Co_ori, Ho, Wo -> N, Co1, Ho, Wo, Co0
    dy = np.pad(dy_ori, ((0, 0), (0, cout - cout_ori), (0, 0), (0, 0)),
                'constant').reshape(batch, cout1, block_size, ho, wo).transpose(0, 1, 3, 4, 2)

    return filter, dy


def _conv_bp_filter(filter_ori, group_dict, block_size=16):
    multi, cin_ori, hk, wk = filter_ori.shape

    real_g = group_dict.get("real_g")
    cin1_g = group_dict.get("cin1_g")
    cin_g = group_dict.get("cin_g")
    cout_g = group_dict.get("cout_g")

    # filter: K, Ci_ori, Hk, Wk -> Co_g, Ci, Hk, Wk -> Ci1, Hk, Wk, Co_g, Ci0
    filter = np.zeros([cout_g, real_g * cin_g, hk, wk], dtype=filter_ori.dtype)
    for n in range(cout_g):
        for c in range(cin_ori):
            if n // multi == c % block_size:
                filter[n, c, :, :] += filter_ori[n % multi, c, :, :]

    filter = filter.reshape(cout_g, real_g * cin1_g, block_size,
                            hk, wk).transpose(1, 3, 4, 0, 2).astype(filter.dtype)

    return filter


@register_input(["depthwise_conv2d_backprop_input"])
def _depthwise_conv2d_backprop_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    weight, dy = context.input_arrays
    filter_ori_format = context.stc_input_ori_formats[0]
    is_gpu = get_global_storage().mode.is_gpu()
    if is_gpu:
        if filter_ori_format == "NCHW":
            weight = weight.transpose(2, 3, 1, 0)
        elif filter_ori_format == "NHWC":
            weight = weight.transpose(1, 2, 3, 0)
        elif filter_ori_format == "HWCN":
            weight = weight.transpose(0, 1, 3, 2)
        return (weight, dy), (weight, dy)
    input_size = context.other_runtime_params.get("input_size")
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    dyn_input_dtypes = context.dyn_input_dtypes
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[0]

    fmap = np.array(input_size, dyn_input_dtypes[0])
    cin_ori = input_size[data_format.index('C')]
    w_dtype = weight.dtype
    filter_format = context.stc_input_formats[0]
    if filter_format == "NC1HWC0":
        k_c1, multi, kh, kw, block_size = weight.shape
        cout_ori = multi * cin_ori
    else:
        filter_ori_format = context.stc_input_ori_formats[0]
        filter_ori_shape = context.stc_ori_inputs[0]
        FILTER_N_ORI_INDEX = filter_ori_format.index("N")
        FILTER_C_ORI_INDEX = filter_ori_format.index("C")
        FILTER_H_ORI_INDEX = filter_ori_format.index("H")
        FILTER_W_ORI_INDEX = filter_ori_format.index("W")
        k_c1 = filter_ori_shape[FILTER_N_ORI_INDEX]
        ori_kc = filter_ori_shape[FILTER_C_ORI_INDEX]
        kh = filter_ori_shape[FILTER_H_ORI_INDEX]
        kw = filter_ori_shape[FILTER_W_ORI_INDEX]
        block_size = weight.shape[-1]
        multi = (ori_kc + block_size - 1) // block_size
        cout_ori = multi * cin_ori

    group_dict = _calculate_group(cin_ori, cout_ori, cin_ori)
    filter_ori = np.random.uniform(low, high, [
        multi, cin_ori, kh, kw]).astype(w_dtype)
    weight = _conv_bp_filter(filter_ori, group_dict, block_size)

    return (fmap, weight, dy), (None, weight, dy)


@register_input(["conv2d_bp_input_transdata"])
@register_input(["conv2d_backprop_input"])
def _conv2d_backprop_input_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    weight, dy = context.input_arrays
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    filter_ori_format = context.stc_input_ori_formats[0]
    is_gpu = get_global_storage().mode.is_gpu()
    if is_gpu:
        if filter_ori_format == "NCHW":
            weight = weight.transpose(2, 3, 1, 0)
        elif filter_ori_format == "NHWC":
            weight = weight.transpose(1, 2, 3, 0)
        return (weight, dy), (weight, dy)
    input_size = context.other_runtime_params.get("input_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    input_formats = context.stc_input_formats
    output_dtype = context.output_dtypes
    output_formats = context.output_formats[0]
    dyn_input_dtypes = context.dyn_input_dtypes
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[0]

    fmap = np.array(input_size, dyn_input_dtypes[0])
    filter_format = context.stc_input_formats[0]
    dy_formats = input_formats[1]
    w_dtype = weight.dtype
    if filter_format == "NC1HWC0":
        cout_ori, k_c1, kh, kw, block_size = weight.shape
    else:
        filter_ori_format = context.stc_input_ori_formats[0]
        filter_ori_shape = context.stc_ori_inputs[0]
        FILTER_N_ORI_INDEX = filter_ori_format.index("N")
        FILTER_C_ORI_INDEX = filter_ori_format.index("C")
        FILTER_H_ORI_INDEX = filter_ori_format.index("H")
        FILTER_W_ORI_INDEX = filter_ori_format.index("W")
        cout_ori = filter_ori_shape[FILTER_N_ORI_INDEX]
        ori_kc = filter_ori_shape[FILTER_C_ORI_INDEX]
        kh = filter_ori_shape[FILTER_H_ORI_INDEX]
        kw = filter_ori_shape[FILTER_W_ORI_INDEX]
        block_size = weight.shape[-1]
        k_c1 = (ori_kc + block_size - 1)//block_size
    cout = _align(cout_ori, block_size)  # filter_n
    cin_ori = input_size[data_format.index('C')]
    if groups == 1:
        w_shape = [k_c1, kh, kw, cout, block_size]  # FRACTAL_Z
        weight = np.random.uniform(
            low, high, w_shape).astype(w_dtype)
    else:
        group_dict = _calculate_group(cin_ori, cout_ori, groups)
        multi = cout_ori // cin_ori
        filter_ori = np.random.uniform(low, high, [
            multi, cin_ori, kh, kw]).astype(w_dtype)
        weight = _conv_bp_filter(filter_ori, group_dict, block_size)

    return (fmap, weight, dy), (None, weight, dy)


@register_input(["conv2d_transpose"])
def _conv2d_transpose_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    ipt_1, weight, ipt_3, ipt_4 = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    dyn_input_dtypes = context.dyn_input_dtypes
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[1]

    ipt_0 = np.array(input_size, dyn_input_dtypes[0])
    w_dtype = weight.dtype
    filter_format = context.stc_input_formats[1]
    if filter_format == "NC1HWC0":
        cout_ori, k_c1, kh, kw, block_size = weight.shape
    else:
        filter_ori_format = context.stc_input_ori_formats[1]
        filter_ori_shape = context.stc_ori_inputs[1]
        FILTER_N_ORI_INDEX = filter_ori_format.index("N")
        FILTER_C_ORI_INDEX = filter_ori_format.index("C")
        FILTER_H_ORI_INDEX = filter_ori_format.index("H")
        FILTER_W_ORI_INDEX = filter_ori_format.index("W")
        cout_ori = filter_ori_shape[FILTER_N_ORI_INDEX]
        ori_kc = filter_ori_shape[FILTER_C_ORI_INDEX]
        kh = filter_ori_shape[FILTER_H_ORI_INDEX]
        kw = filter_ori_shape[FILTER_W_ORI_INDEX]
        block_size = weight.shape[-1]
        k_c1 = (ori_kc + block_size - 1) // block_size
    cout = _align(cout_ori, block_size)  # filter_n
    cin_ori = input_size[data_format.index('C')]
    if groups == 1:
        w_shape = [k_c1, kh, kw, cout, block_size]  # FRACTAL_Z
        weight = np.random.uniform(
            low, high, w_shape).astype(w_dtype)
    else:
        group_dict = _calculate_group(cin_ori, cout_ori, groups)
        multi = cout_ori // cin_ori
        filter_ori = np.random.uniform(low, high, [
            multi, cin_ori, kh, kw]).astype(w_dtype)
        weight = _conv_bp_filter(filter_ori, group_dict, block_size)

    return (ipt_0, ipt_1, weight, ipt_3, ipt_4), (None, ipt_1, weight, ipt_3, ipt_4)


@register_input(["deconvolution"])
def _deconvolution_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    ipt_0, weight, ipt_2, ipt_3 = context.input_arrays
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    ori_shapes = context.stc_ori_inputs
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[1]

    w_dtype = weight.dtype
    filter_format = context.stc_input_formats[1]
    if filter_format == "NC1HWC0":
        cout_ori, k_c1, kh, kw, block_size = weight.shape
    else:
        filter_ori_format = context.stc_input_ori_formats[1]
        filter_ori_shape = context.stc_ori_inputs[1]
        FILTER_N_ORI_INDEX = filter_ori_format.index("N")
        FILTER_C_ORI_INDEX = filter_ori_format.index("C")
        FILTER_H_ORI_INDEX = filter_ori_format.index("H")
        FILTER_W_ORI_INDEX = filter_ori_format.index("W")
        cout_ori = filter_ori_shape[FILTER_N_ORI_INDEX]
        ori_kc = filter_ori_shape[FILTER_C_ORI_INDEX]
        kh = filter_ori_shape[FILTER_H_ORI_INDEX]
        kw = filter_ori_shape[FILTER_W_ORI_INDEX]
        block_size = weight.shape[-1]
        k_c1 = (ori_kc + block_size - 1) // block_size
    cout = _align(cout_ori, block_size)  # filter_n
    w_ori_shape = ori_shapes[1]
    cin_ori = w_ori_shape[data_format.index('C')] * groups

    if groups == 1:
        w_shape = [k_c1, kh, kw, cout, block_size]  # FRACTAL_Z
        weight = np.random.uniform(
            low, high, w_shape).astype(w_dtype)
    else:
        group_dict = _calculate_group(cin_ori, cout_ori, groups)
        multi = cout_ori // cin_ori
        filter_ori = np.random.uniform(low, high, [
            multi, cin_ori, kh, kw]).astype(w_dtype)
        weight = _conv_bp_filter(filter_ori, group_dict, block_size)

    return (ipt_0, weight, ipt_2, ipt_3), (ipt_0, weight, ipt_2, ipt_3)

@register_input(["conv2d_backprop_filter","depthwise_conv2d_backprop_filter"])
def _conv2d_backprop_filter_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    ipt_0, ipt_2 = context.input_arrays
    if is_gpu:
        filter_size = context.other_runtime_params.get("filter_sizes")
    else:
        filter_size = context.other_runtime_params.get("filter_size")
    dyn_input_dtypes = context.dyn_input_dtypes

    ipt_1 = np.array(filter_size, dyn_input_dtypes[1])
    if is_gpu:
        return (ipt_0, ipt_1, ipt_2), (ipt_0, ipt_2)
    else:
        return (ipt_0, ipt_1, ipt_2), (ipt_0, None, ipt_2)

@register_input(["conv2d_backprop_input_drelu"])
def _conv2d_backprop_input_drelu_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    weight, dy = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    dyn_input_dtypes = context.dyn_input_dtypes
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[0]

    fmap = np.array(input_size, dyn_input_dtypes[0])
    filter_format = context.stc_input_formats[0]
    w_dtype = weight.dtype
    if filter_format == "NC1HWC0":
        cout_ori, k_c1, kh, kw, block_size = weight.shape
    else:
        filter_ori_format = context.stc_input_ori_formats[0]
        filter_ori_shape = context.stc_ori_inputs[0]
        FILTER_N_ORI_INDEX = filter_ori_format.index("N")
        FILTER_C_ORI_INDEX = filter_ori_format.index("C")
        FILTER_H_ORI_INDEX = filter_ori_format.index("H")
        FILTER_W_ORI_INDEX = filter_ori_format.index("W")
        cout_ori = filter_ori_shape[FILTER_N_ORI_INDEX]
        ori_kc = filter_ori_shape[FILTER_C_ORI_INDEX]
        kh = filter_ori_shape[FILTER_H_ORI_INDEX]
        kw = filter_ori_shape[FILTER_W_ORI_INDEX]
        block_size = weight.shape[-1]
        k_c1 = (ori_kc + block_size - 1) // block_size

    cout = _align(cout_ori, block_size)  # filter_n
    if data_format == 'NCHW':
        n, cin_ori, h, w = input_size
    else:
        n, h, w, cin_ori = input_size

    if groups == 1:
        w_shape = [k_c1, kh, kw, cout, block_size]  # FRACTAL_Z
        weight = np.random.uniform(
            low, high, w_shape).astype(w_dtype)
    else:
        group_dict = _calculate_group(cin_ori, cout_ori, groups)
        multi = cout_ori // cin_ori
        filter_ori = np.random.uniform(low, high, [
            multi, cin_ori, kh, kw]).astype(w_dtype)
        weight = _conv_bp_filter(filter_ori, group_dict, block_size)

    c1 = (cin_ori + 15) // 16
    input_mask_shape = (n, c1, h, w, 2)
    input_mask = np.random.randint(0, 2, input_mask_shape).astype("uint8")

    return (fmap, weight, dy, input_mask), (None, weight, dy, input_mask)


@register_input(["conv2d_backprop_input_vadd_drelu"])
def _conv2d_backprop_input_vadd_drelu_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    weight, dy = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    output_dtype = context.output_dtypes
    dyn_input_dtypes = context.dyn_input_dtypes
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[0]

    fmap = np.array(input_size, dyn_input_dtypes[0])
    filter_format = context.stc_input_formats[0]
    w_dtype = weight.dtype
    if filter_format == "NC1HWC0":
        cout_ori, k_c1, kh, kw, block_size = weight.shape
    else:
        filter_ori_format = context.stc_input_ori_formats[0]
        filter_ori_shape = context.stc_ori_inputs[0]
        FILTER_N_ORI_INDEX = filter_ori_format.index("N")
        FILTER_C_ORI_INDEX = filter_ori_format.index("C")
        FILTER_H_ORI_INDEX = filter_ori_format.index("H")
        FILTER_W_ORI_INDEX = filter_ori_format.index("W")
        cout_ori = filter_ori_shape[FILTER_N_ORI_INDEX]
        ori_kc = filter_ori_shape[FILTER_C_ORI_INDEX]
        kh = filter_ori_shape[FILTER_H_ORI_INDEX]
        kw = filter_ori_shape[FILTER_W_ORI_INDEX]
        block_size = weight.shape[-1]
        k_c1 = (ori_kc + block_size - 1) // block_size
    cout = _align(cout_ori, block_size)  # filter_n
    if data_format == 'NCHW':
        n, cin_ori, h, w = input_size
    else:
        n, h, w, cin_ori = input_size

    if groups == 1:
        w_shape = [k_c1, kh, kw, cout, block_size]  # FRACTAL_Z
        weight = np.random.uniform(
            low, high, w_shape).astype(w_dtype)
    else:
        group_dict = _calculate_group(cin_ori, cout_ori, groups)
        multi = cout_ori // cin_ori
        filter_ori = np.random.uniform(low, high, [
            multi, cin_ori, kh, kw]).astype(w_dtype)
        weight = _conv_bp_filter(filter_ori, group_dict, block_size)

    c1 = (cin_ori + 15) // 16
    input_mask_shape = (n, c1, h, w, 2)
    input_mask = np.random.randint(0, 2, input_mask_shape).astype("uint8")

    vadd_shape = (n, c1, h, w, 16)
    vadd_data = np.random.uniform(low, high, vadd_shape).astype(output_dtype[0])

    return (fmap, weight, dy, vadd_data, input_mask), (None, weight, dy, vadd_data, input_mask)
