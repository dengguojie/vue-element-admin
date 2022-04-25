#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Conv3d input tensor generator"""
# Standard Packages
from typing import Tuple
from typing import Optional
from ...utilities import get_global_storage

# Third-Party Packages
import numpy
import numpy as np
from .registry import register_input


def _ceil(a, b):
    return (a + b - 1) // b


def _align(a, b):
    return _ceil(a, b) * b


def _lcm(param1, param2):
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, (param1 % param2)

    return temp // param2


def _calculate_group(cin, cout, groups):
    block_size = 16

    mag_factor0 = _lcm(cin // groups, block_size) // (cin // groups)
    mag_factor1 = _lcm(cout // groups, block_size) // (cout // groups)
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin1_g = _align(mag_factor * (cin // groups), block_size)
    cout_g = _align(mag_factor * (cout // groups), block_size)

    group_dict = {"real_g": _ceil(groups, mag_factor),
                  "mag_factor": mag_factor,
                  "cin1_g": cin1_g // block_size,
                  "cout1_g": cout_g // block_size,
                  "groups": groups,
                  "cin_ori": cin // groups,
                  "cout_ori": cout // groups}
    return group_dict


@register_input(["conv3d"])
def _conv3d_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if is_gpu:
        return context.input_arrays,context.input_arrays
    # get OriShape
    block_size = 16
    ori_shapes = context.stc_ori_inputs
    fmap_ori, filter_ori, _, _ = ori_shapes
    ori_formats = context.stc_input_ori_formats
    fmap_ori_format, filter_ori_format, _, _ = ori_formats
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[1]

    fmap_cin_index = fmap_ori_format.index("C")
    filter_cout_index = filter_ori_format.index("N")
    fmap_cin = fmap_ori[fmap_cin_index]
    filter_cout = filter_ori[filter_cout_index]
    
    # Get Group
    groups = context.other_runtime_params.get('groups', 1)
    group_dict = _calculate_group(fmap_cin, filter_cout, groups)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout1_g = group_dict["cout1_g"]
    cin_g = cin1_g * block_size
    cout_g = cout1_g * block_size
    mag_factor = group_dict["mag_factor"]

    # Process Group for Filter : Change to Fractal Z
    # Fmap is NDC1HWC0, Filter is [D, C1, H, W, N, C0]
    fmap = context.input_arrays[0]
    # Remove Dirty Data in Feature Map (A Really Time Consuming Operation)
    for c1_idx in range(fmap.shape[2]): # C1 size
        for c0_idx in range(fmap.shape[-1]): # C0 size
            c_idx = c1_idx * fmap.shape[-1] + c0_idx
            if (c_idx >= fmap_cin):
                fmap[:, :, c1_idx, :, :, c0_idx] = 0

    filter_shape = list(context.input_arrays[1].shape)
    w_d, _, w_h, w_w, _, c0 = filter_shape
    shape_w_frac_z = (real_g, w_d, cin1_g, w_h, w_w, cout_g, c0)
    weight = np.zeros(shape_w_frac_z, dtype=np.float16)
    filter_c = fmap_cin // groups
    filter_ncdhw = (filter_cout, filter_c, w_d, w_h, w_w)
    filter_data = numpy.random.uniform(low, high, filter_ncdhw).astype(context.input_arrays[1].dtype)
    for g in range(groups):
        for ci in range(filter_c):
            for co in range(filter_cout // groups):
                e = g % mag_factor
                dst_cin = e * filter_c + ci
                dst_cout = e * (filter_cout // groups) + co
                src_cout = g * (filter_cout // groups) + co
                weight[g // mag_factor, :, dst_cin // block_size, :, :, dst_cout, dst_cin % block_size] = filter_data[src_cout, ci, :, :, :]

    bias = context.input_arrays[2]
    ipt_3 = context.input_arrays[3]
    if bias is not None:
        bias_shape = list(context.input_arrays[2].shape)
        # bias_shape[0] = (bias_shape[0] + 15) // 16 * 16
        # bias = numpy.random.uniform(low, high, bias_shape).astype(context.input_arrays[2].dtype)
        # set 0 from bias_shape[0] to (bias_shape[0] + 15) // 16 * 16
        bias = numpy.pad(bias, (0, ((bias_shape[0] + 15) // 16 * 16 - bias_shape[0])), constant_values=(0, 0))
    # previous Dynamic/Golden, later static
    print("fmap type = ", fmap.dtype)
    print("weight type = ", weight.dtype)

    return (fmap, weight, bias, ipt_3), (fmap, weight, bias, ipt_3)


@register_input(["conv3d_backprop_input"])
def _conv3d_backprop_input_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if is_gpu:
        return context.input_arrays,context.input_arrays
    inputsize = np.array(context.stc_ori_outputs[0], 'int32')
    block_size = 16
    filter, dy = context.input_arrays
    ori_shapes = context.stc_ori_inputs
    filter_ori_shape, dy_ori_shape = ori_shapes
    ori_formats = context.stc_input_ori_formats
    filter_ori_format, dy_ori_format = ori_formats
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[1]

    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    cout_ori = filter_ori_shape[filter_ori_format.index("N")]
    w_c_ori = filter_ori_shape[filter_ori_format.index("C")]
    cin_ori = context.other_runtime_params.get("input_size")[data_format.index("C")]
    if dy_ori_format == "NDHWC":
        dy_n, dy_d, dy_h, dy_w, dy_c = dy_ori_shape
    elif dy_ori_format == "NCDHW":
        dy_n, dy_c, dy_d, dy_h, dy_w = dy_ori_shape
    else:
        raise RuntimeError("dy format should be NCDHW or NDHWC")

    dy_n, dy_d, cout1, dy_h, dy_w, c0 = dy.shape
    w_d, _, w_h, w_w, _ , _= filter.shape

    groups = context.other_runtime_params.get('groups', 1)
    group_dict = _calculate_group(cin_ori, cout_ori, groups)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout1_g = group_dict["cout1_g"]
    cin_g = cin1_g * block_size
    cout_g = cout1_g * block_size
    mag_factor = group_dict["mag_factor"]
    print("##########################group_dict",group_dict)
    filter_dhwcn_shape = [w_d, w_h, w_w, w_c_ori, cout_ori]
    filter_data = np.random.uniform(low, high, filter_dhwcn_shape).astype(filter.dtype)
    weight_group = np.zeros((real_g, w_d, cin1_g, w_h, w_w, cout_g, c0), dtype=np.float16)
    dy_data = np.random.uniform(low, high, [dy_n, dy_d, dy_h, dy_w, dy_c]).astype(dy.dtype)
    dy_c_align = _align(dy_c, block_size)
    # dy: NDHWC -> NDC1HWC0
    dy = np.pad(dy_data, ((0, 0), (0, 0), (0, 0), (0, 0), (0, dy_c_align - dy_c)), 'constant').reshape(dy_n, dy_d, dy_h, dy_w, dy_c_align//block_size, block_size).transpose(0,1,4,2,3,5)
    # filter : dhwcn -> gdc1hw,n,c0
    for g in range(groups):
        for ci in range(w_c_ori):
            for co in range(cout_ori // groups):
                e = g % mag_factor
                dst_cin = e * w_c_ori + ci
                dst_cout = e * (cout_ori // groups) + co
                src_cout = g * (cout_ori // groups) + co
                weight_group[g // mag_factor, :, dst_cin // block_size, :, :, dst_cout, dst_cin % block_size] = filter_data[:, :, :, ci, src_cout]
    return (inputsize, weight_group, dy), (weight_group, dy)


@register_input(["conv3d_backprop_filter"])
def _conv3d_backprop_filter_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    filtersize = numpy.array(context.stc_ori_outputs[0], 'int32')
    ipt_0 = context.input_arrays[0]
    ipt_1 = context.input_arrays[1]
    is_gpu = get_global_storage().mode.is_gpu()
    if is_gpu:
        return (ipt_0, filtersize, ipt_1), (ipt_0, filtersize, ipt_1)
    else:
        return (ipt_0, filtersize, ipt_1), (ipt_0, ipt_1)


@register_input(["conv3d_transpose"])
def _conv3d_transpose_input(context: "tbetoolkits.UniversalTestcaseStructure"):

    inputsize = np.array(context.stc_ori_outputs[0], 'int32')
    block_size = 16
    dy,filter ,bias,offset_w= context.input_arrays

    ori_shapes = context.stc_ori_inputs
    dy_ori_shape, filter_ori_shape, _, _ = ori_shapes
    ori_formats = context.stc_input_ori_formats
    dy_ori_format, filter_ori_format, _, _ = ori_formats
    input_data_ranges = context.actual_input_data_ranges
    low, high = input_data_ranges[1]

    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    cout_ori = filter_ori_shape[filter_ori_format.index("N")]
    w_c_ori = filter_ori_shape[filter_ori_format.index("C")]
    cin_ori = context.other_runtime_params.get("input_size")[data_format.index("C")]
    if dy_ori_format == "NDHWC":
        dy_n, dy_d, dy_h, dy_w, dy_c = dy_ori_shape
    elif dy_ori_format == "NCDHW":
        dy_n, dy_c, dy_d, dy_h, dy_w = dy_ori_shape
    else:
        raise RuntimeError("dy format should be NCDHW or NDHWC")

    dy_n, dy_d, cout1, dy_h, dy_w, c0 = dy.shape
    w_d, _, w_h, w_w, _ , _= filter.shape
    groups = context.other_runtime_params.get('groups', 1)
    group_dict = _calculate_group(cin_ori, cout_ori, groups)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout1_g = group_dict["cout1_g"]
    cin_g = cin1_g * block_size
    cout_g = cout1_g * block_size
    mag_factor = group_dict["mag_factor"]
    print("##########################group_dict",group_dict)
    filter_dhwcn_shape = [w_d, w_h, w_w, w_c_ori, cout_ori]
    filter_data = np.random.uniform(low, high, filter_dhwcn_shape).astype(filter.dtype)
    weight_group = np.zeros((real_g, w_d, cin1_g, w_h, w_w, cout_g, c0), dtype=np.float16)
    dy_data = np.random.uniform(low, high, [dy_n, dy_d, dy_h, dy_w, dy_c]).astype(dy.dtype)
    dy_c_align = _align(dy_c, block_size)
    # dy: NDHWC -> NDC1HWC0
    dy = np.pad(dy_data, ((0, 0), (0, 0), (0, 0), (0, 0), (0, dy_c_align - dy_c)), 'constant').reshape(dy_n, dy_d, dy_h, dy_w, dy_c_align//block_size, block_size).transpose(0,1,4,2,3,5)
    # filter : dhwcn -> gdc1hw,n,c0
    for g in range(groups):
        for ci in range(w_c_ori):
            for co in range(cout_ori // groups):
                e = g % mag_factor
                dst_cin = e * w_c_ori + ci
                dst_cout = e * (cout_ori // groups) + co
                src_cout = g * (cout_ori // groups) + co
                weight_group[g // mag_factor, :, dst_cin // block_size, :, :, dst_cout, dst_cin % block_size] = filter_data[:, :, :, ci, src_cout]
    print("cuiwen1-------------",dy.shape,weight_group.shape)
    return (inputsize, dy, weight_group,bias,offset_w ), (None,dy, weight_group,bias,offset_w)


@register_input(["avg_pool3d_grad"])
def _avg_pool3d_grad_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    print("================ GET IN AVGPOOL3DGRAD INPUT ===================")

    orig_input_shape = context.other_runtime_params.get("orig_input_shape")
    orig_input_shape_data = numpy.array(context.stc_ori_outputs[0], 'int32')

    ksize = context.other_runtime_params.get("ksize")
    grads_shape = list(context.input_arrays[0].shape)
    mul_shape = list(context.input_arrays[2].shape)

    data_format = context.other_runtime_params.get('data_format', "NDHWC")
    if data_format == 'NDHWC':
        _, FD, FH, FW, FC = orig_input_shape
        _, KD, KH, KW, _ = ksize
        _, stride_d, stride_h, stride_w, _ = context.other_runtime_params.get("strides")
    else:
        # NCDHW
        _, FC, FD, FH, FW = orig_input_shape
        _, _, KD, KH, KW = ksize
        _, _, stride_d, stride_h, stride_w = context.other_runtime_params.get("strides")

    GN, GD, GC1, GH, GW, C0 = grads_shape
    filter_data = numpy.zeros((KD * KH * KW * GC1, 1, 16, 16), dtype=numpy.float16)
    filter_data[:, 0, :, :] = numpy.identity(16)
    # print("filter_data:", filter_data)
    pads = context.other_runtime_params.get("pads")
    mean_matrix = []
    # tf has not support ceil_mode, override, include_pad params
    mean_matrix = numpy.zeros(mul_shape, dtype=numpy.float16)
    if pads == [0,0,0,0,0,0]:
        rate = 1.0 / (KD * KH * KW)
        mean_matrix = numpy.full(mul_shape, rate, dtype=numpy.float16)
    else:
        for n in range(mul_shape[0]):
            for d in range(mul_shape[1]):
                for c1 in range(mul_shape[2]):
                    for h in range(mul_shape[3]):
                        for w in range(mul_shape[4]):
                            for c0 in range(mul_shape[5]):
                                valid_d = min(d * stride_d + KD, pads[0] + FD) - max(pads[0], d * stride_d)
                                valid_h = min(h * stride_h + KH, pads[2] + FH) - max(pads[2], h * stride_h)
                                valid_w = min(w * stride_w + KW, pads[4] + FW) - max(pads[4], w * stride_w)
                                mean_matrix[n][d][c1][h][w][c0] = 1.0 / (valid_d * valid_h * valid_w)
    # print("mean_matrix:", mean_matrix)
    return (orig_input_shape_data, context.input_arrays[0], filter_data), (context.input_arrays[0], filter_data, mean_matrix)


@register_input(["avg_pool3d"])
def _avg_pool3d_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if is_gpu:
        return (context.input_arrays[0],),(context.input_arrays[0],)
    print("================ GET IN AVGPOOL3D INPUT ===================")
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    pads = context.other_runtime_params.get("pads")
    
    fmap_shape = list(context.input_arrays[0].shape)
    mul_shape = list(context.input_arrays[2].shape)
    print("mul_shape:", mul_shape)
    
    d_idx, h_idx, w_idx = data_format.index("D"), data_format.index("H"), data_format.index("W")
    kd, kh, kw = ksize[d_idx], ksize[h_idx], ksize[w_idx]
    sd, sh, sw = strides[d_idx], strides[h_idx], strides[w_idx]
    fn, fd, fc1, fh, fw, C0 = fmap_shape
    
    filter_data = numpy.zeros((kd * kh * kw * fc1, 1, 16, 16), dtype=numpy.float16)
    filter_data[:, 0, :, :] = numpy.identity(16)
    
    mean_matrix = numpy.zeros(mul_shape, dtype=numpy.float16)
    if all(i == 0 for i in pads):
        rate = 1.0 / (kd * kh * kw)
        mean_matrix = numpy.full(mul_shape, rate, dtype=numpy.float16)
    else:
        for n in range(mul_shape[0]):
            for d in range(mul_shape[1]):
                for c1 in range(mul_shape[2]):
                    for h in range(mul_shape[3]):
                        for w in range(mul_shape[4]):
                            for c0 in range(mul_shape[5]):
                                valid_d = min(d * sd + kd, pads[0] + fd) - max(pads[0], d * sd)
                                valid_h = min(h * sh + kh, pads[2] + fh) - max(pads[2], h * sh)
                                valid_w = min(w * sw + kw, pads[4] + fw) - max(pads[4], w * sw)
                                mean_matrix[n][d][c1][h][w][c0] = 1.0 / (valid_d * valid_h * valid_w)
    # print("mean_matrix:", mean_matrix)
    return (context.input_arrays[0], filter_data), (context.input_arrays[0], filter_data, mean_matrix)

