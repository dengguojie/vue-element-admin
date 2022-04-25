#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Conv2d input tensor generator"""
# Standard Packages
from typing import Tuple
from typing import Optional
from ...utilities import get_global_storage
# Third-Party Packages
import numpy
from .registry import register_input


def _avgpool_v2_input(_,
                      __,
                      input_arrays: Tuple[Optional[numpy.ndarray]],
                      ___,
                      low,
                      high) -> Tuple[Tuple[Optional[numpy.ndarray], ...],
                                     Tuple[Optional[numpy.ndarray], ...]]:
    ipt_0 = input_arrays[0]
    shape = list(input_arrays[1].shape)
    Co1 = (shape[0] + 15) // 16
    _, _, H, W, _ = shape
    ipt_1 = numpy.zeros((H * W * Co1, 1, 16, 16), dtype=numpy.float16)
    ipt_1[:, 0, :, :] = numpy.identity(16)
    return (ipt_0, ipt_1), (ipt_0, ipt_1)


@register_input(["avg_pool"])
def _avgpool_mul_input(context: "tbetoolkits.core_modules.dynamic_shape.ProfilingContextStructure"):
    ipt_0 = context.input_arrays[0]
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    padding = context.other_runtime_params.get("padding")
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    output_dtype = context.output_dtypes
    dyn_input_dtypes = context.dyn_input_dtypes
    ipt_1 = context.input_arrays[1]
    ipt_2 = context.input_arrays[2]
    if context.input_arrays[1] is None and context.input_arrays[2] is None:
        return (ipt_0, None, None), (ipt_0, None, None, None)
    if len(context.input_arrays[1].shape) == 4:
        return (ipt_0, ipt_1, None), (ipt_0, ipt_1, None, None)

    shape = list(ipt_1.shape)
    Co, _, H, W, _ = shape
    Co1 = (Co + 15) // 16
    ipt_1 = numpy.zeros((H * W * Co1, 1, 16, 16), dtype=ipt_1.dtype)
    ipt_1[:, 0, :, :] = numpy.identity(16)
    shape = list(ipt_2.shape)
    n, c1, m, c0 = shape #mul_shape: [N,Co1,Ho*Wo,C0]
    _, _, in_size_h, in_size_w, _ = list(ipt_0.shape)
    avg_mean_factor = []
    h_index, w_index = data_format.index("H"), data_format.index("W")
    stride_h, stride_w = strides[h_index], strides[w_index]
    window_h, window_w = ksize[h_index], ksize[w_index]
    if padding == "SAME":
        _, _, Hi, Wi, _ = ipt_0.shape
        ho=(Hi + stride_h - 1) // stride_h
        wo=(Wi + stride_w - 1) // stride_w
        pad_rows = max(0, (ho - 1) * stride_h + window_h - in_size_h)
        pad_cols = max(0, (wo - 1) * stride_w + window_w - in_size_w)
        padT = pad_rows // 2
        padB = pad_rows - padT
        padL = pad_cols // 2
        padR = pad_cols - padL
        avg_mean_factor = []
        for i in range(ho):
            for j in range(wo):
                h_start = i * stride_h - padT
                w_start = j * stride_w - padL
                h_end = min(h_start + window_h, in_size_h)
                w_end = min(w_start + window_w, in_size_w)
                h_start = max(h_start, 0)
                w_start = max(w_start, 0)
                area = max((h_end - h_start) * (w_end - w_start), 1)
                mean_value = int(area)
                avg_mean_factor.append(mean_value)
        avg_mean_factor = numpy.array(avg_mean_factor).astype(output_dtype[0])
    elif padding == "VALID":
        avg_coeff = int(window_h * window_w)
        avg_mean_factor = numpy.random.uniform(avg_coeff, avg_coeff, (m,)).astype(output_dtype[0])
    ipt_2 = numpy.zeros((n, c1, m, c0)).astype(output_dtype[0])
    for i in range(c0):
        ipt_2[:, :, :, i] = avg_mean_factor[:]
    return (ipt_0, ipt_1, None), (ipt_0, ipt_1, ipt_2, None)


@register_input(["avg_pool_v2"])
def _avgpool_v2_mul_input(context: "tbetoolkits.core_modules.dynamic_shape.ProfilingContextStructure"):
    ipt_0, ipt_1, ipt_2= context.input_arrays
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    padding = context.other_runtime_params.get("padding")
    pads = context.other_runtime_params.get("pads")
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    global_pooling = context.other_runtime_params.get["global_pooling"]
    ceil_mode = context.other_runtime_params.get["ceil_mode"]
    output_dtype = context.output_dtypes
    dyn_input_dtypes = context.dyn_input_dtypes
        
    shape = list(ipt_1.shape)
    Co, _, H, W, _ = shape
    Co1 = (Co + 15) // 16
    ipt_1 = numpy.zeros((H * W * Co1, 1, 16, 16), dtype=ipt_1.dtype)
    ipt_1[:, 0, :, :] = numpy.identity(16)
    shape = list(ipt_2.shape)
    n, c1, m, c0 = shape #mul_shape: [N,Co1,Ho*Wo,C0]
    _, _, in_size_h, in_size_w, _ = list(ipt_0.shape)
    avg_mean_factor = []
    h_index, w_index = data_format.index("H"), data_format.index("W")
    stride_h, stride_w = strides[h_index], strides[w_index]
    window_h, window_w = ksize[h_index], ksize[w_index]
    
    _, _, Hi, Wi, _ = ipt_0.shape
    if global_pooling:# or (window_h >= Hi and window_w >= Wi)
        window_h = Hi
        window_w = Wi
        padding = "VALID"
    if padding == "VALID":
        avg_coeff = 1.0 / (window_h * window_w)
        avg_mean_factor = numpy.random.uniform(avg_coeff, avg_coeff, (m,)).astype(numpy.float16)
    else:
        if padding == "SAME":
            ho=(Hi + stride_h - 1) // stride_h
            wo=(Wi + stride_w - 1) // stride_w
            pad_rows = max(0, (ho - 1) * stride_h + window_h - in_size_h)
            pad_cols = max(0, (wo - 1) * stride_w + window_w - in_size_w)
            padT = pad_rows // 2
            padB = pad_rows - padT
            padL = pad_cols // 2
            padR = pad_cols - padL
        else:
            padT, padB, padL, padR = pads
            if ceil_mode:
                Ho = (Hi - window_h + padT + padB + stride_h - 1) // stride_h + 1
                wo = (Wi - window_w + padL + padR + stride_w - 1) // strstride_widew + 1
                # padB = max(0, (ho - 1) * stride_h + window_h - Hi - padT)
                # padR = max(0, (wo - 1) * stride_w + window_w - Wi - padL)
            else:
                ho = (Hi - window_h + padT + padB) // stride_h + 1
                wo = (Wi - window_w + padL + padR) // stride_w + 1
                # padB = max(0, (ho - 1) * stride_h + window_h - Hi - padT)
                # padR = max(0, (wo - 1) * stride_w + window_w - Wi - padL)
        avg_mean_factor = []
        for i in range(ho):
            for j in range(wo):
                h_start = i * stride_h - padT
                w_start = j * stride_w - padL
                h_end = min(h_start + window_h, in_size_h)
                w_end = min(w_start + window_w, in_size_w)
                h_start = max(h_start, 0)
                w_start = max(w_start, 0)
                area = max((h_end - h_start) * (w_end - w_start), 1)
                mean_value = int(area)
                avg_mean_factor.append(mean_value)
        avg_mean_factor = numpy.array(avg_mean_factor).astype(output_dtype[0])

    ipt_2 = numpy.zeros((n, c1, m, c0)).astype(output_dtype[0])
    for i in range(c0):
        ipt_2[:, :, :, i] = avg_mean_factor[:]
    return (ipt_0, ipt_1), (ipt_0, ipt_1, ipt_2)

@register_input(["avg_pool_grad"])
def _avgpoolgrad_input(context: "tbetoolkits.core_modules.dynamic_shape.ProfilingContextStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if is_gpu:
        return (context.input_arrays[0],),(context.input_arrays[0],)
    input_grad, mean_matrix, kernel_matrix= context.input_arrays
    orig_input_shape = context.other_runtime_params.get("orig_input_shape")
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    padding = context.other_runtime_params.get("padding")
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    output_dtype = context.output_dtypes
    dyn_input_dtypes = context.dyn_input_dtypes

    n, dy_c1, dy_h, dy_w, c0 = list(input_grad.shape)
    filter_format = context.stc_input_formats[2]
    if filter_format == "NC1HWC0":
        dy_c, dx_c1, kh, kw, c0 = list(kernel_matrix.shape)
        # transfer into placeholder shape
        kernel_matrix = numpy.zeros((dy_c1, kh*kw, 1, c0, c0), dtype=kernel_matrix.dtype)
        kernel_matrix[:, :, 0, :, :] = numpy.identity(16)
        kernel_matrix = kernel_matrix.astype(kernel_matrix.dtype)
    else:
        # transfer into placeholder shape
        kernel_matrix = numpy.zeros(kernel_matrix.shape, dtype=kernel_matrix.dtype)
        kernel_matrix[:, 0, :, :] = numpy.identity(16)
        kernel_matrix = kernel_matrix.astype(kernel_matrix.dtype)
 
    h_index, w_index = data_format.index("H"), data_format.index("W")
    stride_h, stride_w = strides[h_index], strides[w_index]
    window_h, window_w = ksize[h_index], ksize[w_index]
    dx_h, dx_w = orig_input_shape[h_index], orig_input_shape[w_index]
    if padding == "SAME":
        avg_mean_factor = numpy.empty((dy_h, dy_w), dtype=input_grad.dtype)
        ho = (dx_h + stride_h - 1) // stride_h
        wo = (dx_w + stride_w - 1) // stride_w
        pad_rows = max(0, (ho - 1) * stride_h + window_h - dx_h)
        pad_cols = max(0, (wo - 1) * stride_w + window_w - dx_w)
        padT = pad_rows // 2
        padB = pad_rows - padT
        padL = pad_cols // 2
        padR = pad_cols - padL
        for i in range(dy_h):
            for j in range(dy_w):
                h_start = i * stride_h - padT
                w_start = j * stride_w - padL
                h_end = min(h_start + window_h, dx_h)
                w_end = min(w_start + window_w, dx_w)
                h_start = max(h_start, 0)
                w_start = max(w_start, 0)
                area = max((h_end - h_start) * (w_end - w_start), 1)
                #mean_value = numpy.float16(area)
                avg_mean_factor[i][j] = area
    elif padding == "VALID":
        avg_coeff = window_h * window_w
        avg_mean_factor = numpy.random.uniform(avg_coeff, avg_coeff, (dy_h, dy_w)).astype(input_grad.dtype)
    avg_mean = numpy.reciprocal(avg_mean_factor)
    mean_matrix = numpy.zeros((n, dy_c1, dy_h, dy_w, c0)).astype(input_grad.dtype)
    for i in range(c0):
        mean_matrix[:, :, :, :, i] = avg_mean[:][:]
    orig_input = numpy.array(orig_input_shape, dyn_input_dtypes[0])

    return (orig_input, input_grad, kernel_matrix), (input_grad, mean_matrix, kernel_matrix)
