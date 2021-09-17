"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

AvgPoolGradD ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("AvgPoolGradD", None, None)

case1 = {"params": [{"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,16),"ori_format": "NHWC"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,16),"ori_format": "NHWC"},
                    {"shape": (1,4,4,1,16,16), "dtype": "float16", "format": "C1HWNCoC0", "ori_shape": (1,4,4,1),"ori_format": "NHWC"},
                    {"shape": (1,1,4,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,4,4,16),"ori_format": "NHWC"},
                    [1,4,4,16], [1,4,4,1], [1,1,1,1], "VALID"],
         "case_name": "AvgPoolGradD_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1,1,540,960,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,540,960,16),"ori_format": "NHWC"},
                    {"shape": (1,1,540,960,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,540,960,16),"ori_format": "NHWC"},
                    {"shape": (1,2,2,1,16,16), "dtype": "float16", "format": "C1HWNCoC0", "ori_shape": (1,2,2,1),"ori_format": "NHWC"},
                    {"shape": (1,1,1080,1920,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1080,1920,16),"ori_format": "NHWC"},
                    [1,1080,1920,16], [1,2,2,1], [1,2,2,1], "VALID"],
         "case_name": "AvgPoolGradD_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,1,1,2048,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,2048,16),"ori_format": "NHWC"},
                    {"shape": (1,1,1,2048,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,2048,16),"ori_format": "NHWC"},
                    {"shape": (1,1,1,1,16,16), "dtype": "float16", "format": "C1HWNCoC0", "ori_shape": (1,1,1,1),"ori_format": "NHWC"},
                    {"shape": (1,1,1,2048,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,2048,16),"ori_format": "NHWC"},
                    [1,1,2048,16], [1,1,1,1], [1,1,1,1], "SAME"],
         "case_name": "AvgPoolGradD_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (7,2,1,70,16), "dtype": "float16", "format": "NHWC", "ori_shape": (7,1,70,31),"ori_format": "NHWC"},
                    {"shape": (7,2,1,70,16), "dtype": "float16", "format": "NHWC", "ori_shape": (7,1,70,31),"ori_format": "NHWC"},
                    {"shape": (1,1,254,1), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,254,1),"ori_format": "NHWC"},
                    {"shape": (7,2,1,811,16), "dtype": "float16", "format": "NHWC", "ori_shape": (7,1,811,31),"ori_format": "NHWC"},
                    [7,1,811,31], [1,1,254,1], [1,1,8,1], "VALID"],
         "case_name": "AvgPoolGradD_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": False}

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)

def _NCHW_to_NC1C0HW(tensor):
    c0 = 16
    dim = list(tensor.shape)
    padding = dim[1] % c0

    if padding != 0:
        d = dim[1]
        dim[1] = dim[1] + c0 - padding
        tensor_pad = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        for i in range(dim[0]):
            tensor_pad[i, 0:d, :, :] = tensor[i, :, :, :]
    else:
        tensor_pad = tensor

    dims = [dim[0], dim[1] // c0, c0, dim[2], dim[3]]
    tensor_pad = tensor_pad.reshape(dims)
    return tensor_pad

def _NCHW_to_NC1HWC0(tensor):
    c0 = 16
    dim = list(tensor.shape)
    padding = dim[1] % c0

    if padding != 0:
        d = dim[1]
        dim[1] = dim[1] + c0 - padding
        tensor_pad = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        for i in range(dim[0]):
            tensor_pad[i, 0:d, :, :] = tensor[i, :, :, :]
    else:
        tensor_pad = tensor

    dims = [dim[0], dim[1] // c0, c0, dim[2], dim[3]]
    tensor_pad = tensor_pad.reshape(dims).transpose(0, 1, 3, 4, 2)
    return tensor_pad

def tf_get_windowed_output_size_verbose_V2(input_size, filter_size,
                                           dilation_rate, stride, padding_type):
    if stride <= 0:
        raise RuntimeError("Stride must be > 0, but got", stride)

    if dilation_rate < 1:
        raise RuntimeError("Dilation rate must be >= 1, but got", dilation_rate)

    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    if padding_type == "VALID":
        output_size = (input_size - effective_filter_size + stride) // stride
        padding_before = 0
        padding_after = 0
    elif padding_type == "SAME":
        output_size = (input_size + stride - 1) // stride
        padding_needed = max(0, ((output_size - 1) * stride + effective_filter_size - input_size))
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    else:
        raise RuntimeError("Unsupported padding type", padding_type)

    return output_size, padding_before, padding_after

def tf_get_windowed_output_size_verbose(input_size, filter_size, stride,
                                        padding_type):
    dilation_rate = 1

    (output_size, padding_before,
     padding_after) = tf_get_windowed_output_size_verbose_V2(
        input_size, filter_size, dilation_rate, stride, padding_type)

    return output_size, padding_before, padding_after

def tf_get_windowed_output_size(input_size, filter_size, stride, padding_type):
    if padding_type == 'EXPLICIT':
        raise RuntimeError(
            "tf_get_windowed_output_size does not handle "
            "EXPLITCIT padding; call tf_get_windowed_output_size_verbose "
            "instead.")

    output_size, padding_size, _ = tf_get_windowed_output_size_verbose(
        input_size, filter_size, stride, padding_type)

    return output_size, padding_size

def average_pooling_forward(x,ksize, strides, padding, data_format):
    out=None
    N,C,H,W=x.shape
    _, HH, WW, _ = ksize
    _, stride_h, stride_w, _ = strides
    Ho, P0, P1 = tf_get_windowed_output_size_verbose(H, HH, stride_h, padding)
    Wo, P2, P3 = tf_get_windowed_output_size_verbose(W, WW, stride_w, padding)
    out=np.zeros((N,C,Ho,Wo))
    x_pad = np.zeros((N, C, H + P0 + P1, W + P2 + P3))
    x_pad[:, :, P0:P0 + H, P2:P2 + W] = x
    for i in range(Ho):
        for j in range(Wo):
            x_mask=x_pad[:,:,i*stride_h:i*stride_h+HH,j*stride_w:j*stride_w+WW]
            out[:,:,i,j]=np.average(x_mask,axis=(2,3))
    return out

def conv_forward_naive(x, w,strides, padding, data_format, dilations, name):
    out = None
    N, C, H, W = x.shape
    _, _, HH, WW = w.shape
    _, stride_h, stride_w, _ = strides
    Ho, P0, P1 = tf_get_windowed_output_size_verbose(H, HH, stride_h, padding)
    Wo, P2, P3 = tf_get_windowed_output_size_verbose(W, WW, stride_w, padding)
    x_pad = np.zeros((N, C, H + P0 + P1, W + P2 + P3))
    x_pad[:, :, P0:P0 + H, P2:P2 + W] = x
    out=np.zeros((N,C,Ho,Wo))
    for f in range(1):
        for i in range(Ho):
            for j in range(Wo):
                out[:, :, i, j] = np.sum(x_pad[:, :, i * stride_h: i * stride_h + HH,
                                         j * stride_w: j * stride_w + WW] * w[0, :, :, :],axis=(2, 3))
    return out

def _conv2d(feature_map, weight, strides=[1, 1, 1, 1], padding=None):
    ish = feature_map.shape
    fsh = weight.shape
    output_h = (ish[1] - fsh[0]) // strides[1] + 1
    output_w = (ish[2] - fsh[1]) // strides[2] + 1
    output = np.zeros([ish[0], output_h, output_w, fsh[3]])
    osh = output.shape

    for p in range(osh[0]):
        for i in range(osh[1]):
            for j in range(osh[2]):
                for di in range(fsh[0]):
                    for dj in range(fsh[1]):
                        t = np.dot(
                            feature_map[p, strides[1] * i + di, strides[2] * j +
                                                                dj, :], weight[di, dj, :, :])
                        output[p, i, j] = np.sum([t, output[p, i, j]], axis=0)
    return output

def depthwise_grad(input_sizes,
                   weight,
                   out_backprop,
                   strides,
                   padding,
                   data_format='NHWC',
                   dilations=[1, 1, 1, 1],
                   name=None):
    batch = input_sizes[0]
    input_height = input_sizes[1]
    input_width = input_sizes[2]
    in_channels = input_sizes[3]
    filter_height = weight.shape[0]
    filter_width = weight.shape[1]
    out_height_orig = out_backprop.shape[1]
    out_width_orig = out_backprop.shape[2]
    out_channels = out_backprop.shape[3]
    stride_h = strides[1]
    stride_w = strides[2]
    dilation_rate = 1


    effective_filter_size = (filter_height - 1) * dilation_rate + 1
    if padding == "SAME":
        output_size_h = (input_height + stride_h - 1) // stride_h
        padding_needed = max(
            0, (output_size_h - 1) * stride_h + effective_filter_size - input_height)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    elif padding == "VALID":
        output_size_h = (input_height - effective_filter_size + stride_h) // stride_h
        padding_before = 0
        padding_after = 0
    out_height = output_size_h
    padding_top = padding_before
    padding_bottom = padding_after

    effective_filter_size = (filter_width - 1) * dilation_rate + 1
    if padding == "SAME":
        output_size_w = (input_width + stride_w - 1) // stride_w
        padding_needed = max(
            0, (output_size_w - 1) * stride_w + effective_filter_size - input_width)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    elif padding == "VALID":
        output_size_w = (input_width - effective_filter_size + stride_w) // stride_w
        padding_before = 0
        padding_after = 0
    out_width = output_size_w
    padding_left = padding_before
    padding_right = padding_after

    # padding and dilation

    padded_dilated_height = input_height + filter_height - 1
    padded_dilated_width = input_width + filter_width - 1
    padding_out_top = filter_height - 1 - padding_top
    padding_out_left = filter_width - 1 - padding_left

    padded_dilated_grad = np.zeros(
        [batch, padded_dilated_height, padded_dilated_width, out_channels])
    for i in range(0, out_height):
        index_h = padding_out_top + i * stride_h
        for j in range(0, out_width):
            index_w = padding_out_left + j * stride_w
            padded_dilated_grad[:, index_h, index_w, :] = out_backprop[:, i,
                                                          j, :]

    filter_rotated = np.zeros([filter_height, filter_width, in_channels, 1])
    for i in range(0, filter_height):
        for j in range(0, filter_width):
            filter_rotated[filter_height - 1 - i, filter_width - 1 -
                                                  j, :, :] = weight[i, j, :, :]

    input_grad = np.zeros([batch, input_height, input_width, in_channels])
    padded_dilated_grad_piece = np.zeros(
        [batch, padded_dilated_height, padded_dilated_width, 1])
    filter_rotated_piece = np.zeros([filter_height, filter_width, 1, 1])
    for i in range(0, in_channels):
        padded_dilated_grad_piece[:, :, :, 0] = padded_dilated_grad[:, :, :, i]
        filter_rotated_piece[:, :, 0, 0] = filter_rotated[:, :, i, 0]
        input_grad[:, :, :, i] = _conv2d(
            padded_dilated_grad_piece, filter_rotated_piece).reshape(
            [batch, input_height, input_width])

    return input_grad

def avg_pool_grad_no_global_data(inputShape, ksize_Shape, dtype,strides, padding, data_format, dilations,
                                 name="tf_avg_pool_grad"):

    inputShape_N, inputShape_C, inputShape_H, inputShape_W = inputShape
    _, ksize_h, ksize_w, _ = ksize_Shape
    stride_h, stride_w = strides
    strides = 1, stride_h, stride_w, 1

    depthwise_grad_kernel_shape = (1, inputShape_C, ksize_h, ksize_w)
    depthwise_grad_kernel_KCHW = np.ones(depthwise_grad_kernel_shape)
    mean_value_x = np.ones(inputShape).astype(np.float32)
    # simulation mean value table, save avg pool mean value. == action of depthwise forward ,
    # input = [1,1,...], filter=[1]
    # conv_forward_naive surport only NCHW
    area_value_NCHW = conv_forward_naive(mean_value_x, depthwise_grad_kernel_KCHW, strides=strides, padding=padding,
                                         data_format="NCHW", dilations=dilations, name=name)
    # simulation mean value table out, shape == avg pool out shape, mean_value_pool_out
    mean_value_table_NCHW = np.reciprocal(area_value_NCHW)

    x1 = np.random.randn(*inputShape).astype(np.float32)
    pool_out_no_mean_NCHW = conv_forward_naive(x1, depthwise_grad_kernel_KCHW, strides=strides, padding=padding,
                                               data_format="NCHW", dilations=dilations, name=name)

    mean_value_table_NCHW = mean_value_table_NCHW.astype(np.float16)
    mean_value_table_NCHW = mean_value_table_NCHW.astype(np.float32)

    pool_out_no_mean_NCHW = np.random.randn(*pool_out_no_mean_NCHW.shape).astype(np.float16) * 1000
    pool_out_no_mean_NCHW = pool_out_no_mean_NCHW.astype(np.float32)
    pool_out_NCHW = np.multiply(mean_value_table_NCHW, pool_out_no_mean_NCHW)

    # simulation by depthwise backward input : generator dilation/inputshape/filter/depthwise forward out
    # simulation depthwise backe weight
    pool_out_NCHW = pool_out_NCHW.astype(np.float16)
    pool_out_NCHW = pool_out_NCHW.astype(np.float32)

    dout_5d = _NCHW_to_NC1HWC0(pool_out_NCHW)
    dout_5d_fp16 = dout_5d.astype(np.float16)

    mean_value_table_NCHW = mean_value_table_NCHW.astype(np.float16)
    mean_value_table_NCHW = mean_value_table_NCHW.astype(np.float32)

    mean_value_pool_out_NCHW = np.multiply(mean_value_table_NCHW, pool_out_NCHW)
    mean_value_pool_out_NCHW = mean_value_pool_out_NCHW.astype(np.float16)
    mean_value_pool_out_NCHW = mean_value_pool_out_NCHW.astype(np.float32)

    # simulation depthwise backward API
    depthwise_grad_kernel_KCHW = depthwise_grad_kernel_KCHW.astype(np.float16)
    depthwise_grad_kernel_KCHW = depthwise_grad_kernel_KCHW.astype(np.float32)
    depthwise_grad_kernel_HWCK = depthwise_grad_kernel_KCHW.transpose(2,3,1,0)

    block_size = 16
    c1 = (inputShape_C + block_size - 1) // block_size
    filter_6d_shape = (c1, 1, ksize_h, ksize_w, block_size, block_size)
    filter_5d = _NCHW_to_NC1C0HW(depthwise_grad_kernel_KCHW).transpose(1, 3, 4, 0, 2)
    filter_6d = np.zeros(filter_6d_shape, dtype=np.float32)
    for d4 in range(block_size):
        for d5 in range(block_size):
            if d4 == d5:
                filter_6d[:, 0, :, :, d4, d5] = filter_5d[:, :, :, 0, d5]

    filter_6d_fp16 = filter_6d.astype(np.float16)

    _, stride_h, stride_w, _ = strides
    inputShape_NHWC = inputShape[0], inputShape[2], inputShape[3], inputShape[1]
    dvaluemean_5d = _NCHW_to_NC1HWC0(mean_value_table_NCHW)
    dvaluemean_5d_fp16 = dvaluemean_5d.astype(np.float16)

    mean_value_pool_out_NHWC = mean_value_pool_out_NCHW.transpose(0,2,3,1)
    simulator_depthwise_grad_input_NHWC = depthwise_grad(inputShape_NHWC, depthwise_grad_kernel_HWCK,
                                                         mean_value_pool_out_NHWC, strides=strides,padding=padding,
                                                         data_format="NHWC", dilations=dilations,name=name)
    simulator_depthwise_backward_input_NCHW = simulator_depthwise_grad_input_NHWC.transpose(0, 3, 1, 2)
    input_grad_5d = _NCHW_to_NC1HWC0(simulator_depthwise_backward_input_NCHW)
    input_grad_5d_fp16 = input_grad_5d.astype(np.float16)
    return input_grad_5d_fp16, simulator_depthwise_backward_input_NCHW, dvaluemean_5d_fp16,mean_value_table_NCHW, \
           filter_6d_fp16, depthwise_grad_kernel_KCHW, dout_5d_fp16, pool_out_NCHW

calc_expect_func_res={}

def case_gen(inputShape, ksizes, strides, dtype="float16", padding="VALID", calc_expect_func=0):
    inputShapeNCHW = inputShape[0], inputShape[3], inputShape[1], inputShape[2]
    strides2 = (strides[1], strides[2])

    input_grad_5d_fp16, simulator_depthwise_backward_input_NCHW, dvaluemean_5d_fp16,mean_value_table_NCHW, \
    filter_6d_fp16, depthwise_grad_kernel_KCHW, dout_5d_fp16, pool_out_NCHW = avg_pool_grad_no_global_data(
        inputShapeNCHW, ksizes, dtype = dtype, strides = strides2,
        padding = padding, data_format="NCHW", dilations = (1,1,1,1),name = "avg_pool_grad")
    def _shape_NCHW_NHWC(shape):
        n,c,h,w = shape
        return (n,h,w,c)
    orig_input_shape = _shape_NCHW_NHWC(simulator_depthwise_backward_input_NCHW.shape)
    calc_expect_func_res[calc_expect_func] = input_grad_5d_fp16
    return {
        "params": [{"shape": dout_5d_fp16.shape, "dtype": "float16", "format": "NC1HWC0",
                    "ori_shape": _shape_NCHW_NHWC(pool_out_NCHW.shape),"ori_format": "NHWC", "param_type": "input", "value":dout_5d_fp16},
                   {"shape": dvaluemean_5d_fp16.shape, "dtype": "float16", "format": "NC1HWC0",
                    "ori_shape": _shape_NCHW_NHWC(mean_value_table_NCHW.shape),"ori_format": "NHWC", "param_type": "input", "value":dvaluemean_5d_fp16},
                   {"shape": filter_6d_fp16.shape, "dtype": "float16", "format": "C1HWNCoC0",
                    "ori_shape": _shape_NCHW_NHWC(depthwise_grad_kernel_KCHW.shape),"ori_format": "NHWC", "param_type": "input", "value":filter_6d_fp16},
                   {"shape": input_grad_5d_fp16.shape, "dtype": "float16", "format": "NC1HWC0",
                    "ori_shape": orig_input_shape,"ori_format": "NHWC", "param_type": "output"},
                   orig_input_shape, ksizes, strides, "SAME"],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    }

def calc_expect_func_1(input_grad, mean_matrix, kernel_matrix, out_grad, orig_input_shape, ksize, strides, padding):
    return calc_expect_func_res[calc_expect_func_1]
case_param_1 = case_gen(inputShape=(1, 1922, 3, 1), ksizes=(1, 4, 1, 1), strides=(1, 2, 2, 1), dtype="float16", padding="SAME", calc_expect_func = calc_expect_func_1)
ut_case.add_precision_case("Ascend910A", case_param_1)

def calc_expect_func_2(input_grad, mean_matrix, kernel_matrix, out_grad, orig_input_shape, ksize, strides, padding):
    return calc_expect_func_res[calc_expect_func_2]
case_param_2 = case_gen(inputShape=(1, 5, 5, 1), ksizes=(1, 3, 3, 1), strides=(1, 2, 2, 1), dtype="float16", padding="SAME", calc_expect_func = calc_expect_func_2)
ut_case.add_precision_case("Ascend910A", case_param_2)

def calc_expect_func_3(input_grad, mean_matrix, kernel_matrix, out_grad, orig_input_shape, ksize, strides, padding):
    return calc_expect_func_res[calc_expect_func_3]
case_param_3 = case_gen(inputShape=(1, 10, 1, 16), ksizes=(1, 5, 1, 1), strides=(1, 2, 2, 1), dtype="float16", padding="SAME", calc_expect_func = calc_expect_func_3)
ut_case.add_precision_case("Ascend910A", case_param_3)

def calc_expect_func_4(input_grad, mean_matrix, kernel_matrix, out_grad, orig_input_shape, ksize, strides, padding):
    return calc_expect_func_res[calc_expect_func_4]
case_param_4 = case_gen(inputShape=(1, 10, 1, 1), ksizes=(1, 5, 1, 1), strides=(1, 2, 2, 1), dtype="float16", padding="SAME", calc_expect_func = calc_expect_func_4)
ut_case.add_precision_case("Ascend910A", case_param_4)

# if __name__ == '__main__':
#     ut_case.run("Ascend910")
