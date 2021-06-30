# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("AvgPoolV2GradD", "impl.avg_pool_v2_grad_d", "avg_pool_v2_grad_d")

def _NCHW_to_NC1HWC0(tensor):
    """
    input tensor is a 4D feature map,
    with a shape [N, C, H, W]
    padding C to C1*C0, where C0 = 16
    output: tensor_pad[N, C1, H, W, C0]
    """
    c0 = 16
    dim = list(tensor.shape)
    padding = dim[1] % c0

    if padding != 0:
        d = dim[1]
        dim[1] = dim[1] + c0 - padding
        tensor_pad = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        for i in range(dim[0]):
            tensor_pad[i, 0:d, : :] = tensor[i, :, :, :]
    else:
        tensor_pad = tensor

    dims = [dim[0], dim[1] // c0, c0, dim[2], dim[3]]
    tensor_pad = tensor_pad.reshape(dims).transpose(0, 1, 3, 4, 2)

    return tensor_pad

def _NC1HWC0_to_NCHW(tensor):
    """
    input tensor is a 5D feature map,
    with a shape [N, C1, H, W, C0]
    padding C to C1*C0, where C0 = 16
    output: tensor_pad[N, C, H, W]
    """
    Ftemp = np.shape(tensor)
    F = [Ftemp[0], Ftemp[1]*Ftemp[4], Ftemp[2], Ftemp[3]]
    outputData = np.zeros(F)
    for i in range(Ftemp[0]):
        for j in range(Ftemp[1]):
            for k in range(Ftemp[4]):
                outputData[i,j*Ftemp[4]+k,:,:] = tensor[i,j,:,:,k]
    return outputData

def _NCHW_to_NC1C0HW(tensor):
    """
    input tensor is a 4D feature map,
    with a dimension [N, C, H, W]
    padding C to C1*C0, where C0 = 16
    output: tensor_pad[N, C1, C0, H, W]
    """
    c0 = 16
    dim = list(tensor.shape)
    padding = dim[1] % c0

    if padding != 0:
        d = dim[1]
        dim[1] = dim[1] + c0 - padding
        tensor_pad = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        for i in range(dim[0]):
            tensor_pad[i, 0:d, : :] = tensor[i, :, :, :]
    else:
        tensor_pad = tensor

    dims = [dim[0], dim[1] // c0, c0, dim[2], dim[3]]
    tensor_pad = tensor_pad.reshape(dims)

    return tensor_pad

def _NHWC_to_NC1HWC0(tensor):
    """
    input tensor is a 4D feature map,
    with a shape [N, H, W, C]
    padding C to C1*C0, where C0 = 16
    output: tensor_pad[N, C1, H, W, C0]
    """
    c0 = 16
    dim = list(tensor.shape)
    padding = dim[3] % c0

    if padding != 0:
        d = dim[3]
        dim[3] = dim[3] + c0 - padding
        tensor_pad = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        for i in range(dim[0]):
            tensor_pad[i, :, :, 0:d] = tensor[i, :, :, :]
    else:
        tensor_pad = tensor

    dims = [dim[0], dim[3] // c0, c0, dim[1], dim[2]]
    tensor_pad = tensor_pad.reshape(dims)

    return tensor_pad

def tf_get_windowed_output_size_verbose_v2(input_size, filter_size, dilation_rate,
                                           stride, padding_type, pad1, pad2, ceil_mode):
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
        padding_needed = max(0, (output_size - 1) * stride + effective_filter_size - input_size)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    elif padding_type == "CALCULATED":
        if ceil_mode:
            output_size = ((input_size - effective_filter_size +
                            pad1 + pad2 + stride - 1) // stride) + 1
        else:
            output_size = ((input_size - effective_filter_size +
                            pad1 + pad2) // stride) + 1
        padding_needed = max(0, (output_size - 1) * stride + effective_filter_size - input_size)
        padding_before = pad1
        padding_after = padding_needed - padding_before
    else:
        raise RuntimeError("Unsupported padding type", padding_type)

    return output_size, padding_before, padding_after

def tf_get_windowed_output_size_verbose(input_size, filter_size, stride,
                                        padding_type, pad1, pad2, ceil_mode):
    dilation_rate = 1

    (output_size, padding_before,
     padding_after) = tf_get_windowed_output_size_verbose_v2(
        input_size, filter_size, dilation_rate, stride, padding_type, pad1, pad2, ceil_mode)

    return output_size, padding_before, padding_after

def conv_forward_naive(x, w, strides, padding, pads, ceil_mode,
                       exclusive):
    out = None
    N, C, H, W = x.shape
    _, _, HH, WW = w.shape
    _, stride_h, stride_w, _ = strides
    Ho, P0, P1 = tf_get_windowed_output_size_verbose(H, HH, stride_h, padding, pads[0], pads[1], ceil_mode)
    Wo, P2, P3 = tf_get_windowed_output_size_verbose(W, WW, stride_w, padding, pads[2], pads[3], ceil_mode)
    x_pad = np.zeros((N, C, H + P0 + P1, W + P2 + P3))
    x_pad[:, :, P0:P0 + H, P2:P2 + W] = x
    out = np.zeros((N,C,Ho,Wo))
    for f in range(1):
        for i in range(Ho):
            for j in range(Wo):
                if exclusive:
                    out[:, :, i, j] = np.sum(x_pad[:, :, i * stride_h: i * stride_h + HH,
                                             j * stride_w: j * stride_w + WW] * w[0, :, :, :], axis=(2,3))
                else:
                    out[:, :, i, j] = HH * WW
    return out

def depthwise_grad(input_sizes, weight, out_backprop, strides, padding, pads, ceil_mode):

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
        padding_needed = max(0, (output_size_h - 1) * stride_h + effective_filter_size - input_height)
        padding_before = padding_needed // 2
    elif padding == "VALID":
        output_size_h = (input_height - effective_filter_size + stride_h) // stride_h
        padding_before = 0
    elif padding == "CALCULATED":
        if ceil_mode:
            output_size_h = ((input_height - effective_filter_size +
                            pads[0] + pads[1] + stride_h - 1) // stride_h) + 1
        else:
            output_size_h = ((input_height - effective_filter_size +
                              pads[0] + pads[1]) // stride_h) + 1
        padding_before = pads[0]

    out_height = output_size_h
    padding_top = padding_before

    effective_filter_size = (filter_width - 1) * dilation_rate + 1
    if padding == "SAME":
        output_size_w = (input_width + stride_w - 1) // stride_w
        padding_needed = max(0, (output_size_w - 1) * stride_w + effective_filter_size - input_width)
        padding_before = padding_needed // 2
    elif padding == "VALID":
        output_size_w = (input_width - effective_filter_size + stride_w) // stride_w
        padding_before = 0
    elif padding == "CALCULATED":
        if ceil_mode:
            output_size_w = ((input_width - effective_filter_size +
                              pads[2] + pads[3] + stride_w - 1) // stride_w) + 1
        else:
            output_size_w = ((input_width - effective_filter_size +
                              pads[2] + pads[3]) // stride_w) + 1
        padding_before = pads[2]
    out_width = output_size_w
    padding_left = padding_before

    # padding and dilation
    padded_dilated_height = input_height + filter_height - 1
    padded_dilated_width = input_width + filter_height - 1
    padding_out_top = filter_height - 1 - padding_top
    padding_out_left = filter_width - 1 - padding_left
    padded_dilated_grad = np.zeros(
        [batch, padded_dilated_height, padded_dilated_width, out_channels])

    for i in range(0, out_height):
        index_h = padding_out_top + i * stride_h
        for j in range(0, out_width):
            index_w = padding_out_left + j * stride_w
            padded_dilated_grad[:, index_h, index_w, :] = out_backprop[:, i, j, :]

    filter_rotated = np.zeros([filter_height, filter_width, in_channels, 1])
    for i in range(0, filter_height):
        for j in range(0, filter_width):
            filter_rotated[filter_height - 1 - i, filter_width - 1 -
                                                  j, :, :] = weight[i , j, :, :]

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

def _conv2d(feature_map, weight, strides=(1,1,1,1)):
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
                    for dj in range(fsh[0]):
                        t = np.dot(
                            feature_map[p, strides[1] * i + di, strides[2] * j +
                                                                dj, :], weight[di, dj, : ,:])
                        output[p, i, j] = np.sum([t, output[p, i, j]], axis=0)
    return output

def gen_input_value(orig_input_shape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode, exclusive):
    ####
    dilations = [1,1,1,1]
    name = None
    inputShape = orig_input_shape
    block_size = 16
    if data_format == "NCHW":
        inputShape_N, inputShape_C, inputShape_H, inputShape_W = inputShape
        _, _, ksize_h, ksize_w = ksize
        _, _, stride_h, stride_w = strides
    elif data_format == "NHWC":
        inputShape_N, inputShape_H, inputShape_W, inputShape_C = inputShape
        _, ksize_h, ksize_w, _ = ksize
        _, stride_h, stride_w, _ = strides
    else:
        raise RuntimeError("only support NCHW NHWC,")
    if global_pooling:
        inputShape_NCHW = inputShape_N, inputShape_C, 1, 1
        pool_out_no_mean_NCHW = np.random.randn(*inputShape_NCHW).astype(np.float32)
        mean_matrix_value = None
        kernel_matrix_value = None
    else:
        # mean_matrix_value
        depthwise_grad_kernel_shape = (1, inputShape_C, ksize_h, ksize_w)
        strides = 1, stride_h, stride_w, 1
        inputShape_NCHW = inputShape_N, inputShape_C, inputShape_H, inputShape_W
        mean_value_x = np.ones(inputShape_NCHW)
        depthwise_grad_kernel_KCHW = np.ones(depthwise_grad_kernel_shape)
        area_value_NCHW = conv_forward_naive(mean_value_x, depthwise_grad_kernel_KCHW, strides,
                                             padding_mode, pads, ceil_mode, exclusive)
        mean_value_table_NCHW = np.reciprocal(area_value_NCHW)
        mean_value_table_NCHW = mean_value_table_NCHW.astype(np.float16)
        mean_value_table_NCHW = mean_value_table_NCHW.astype(np.float32)
        mean_matrix_value = _NCHW_to_NC1HWC0(mean_value_table_NCHW)
        # kernel_matrix_value
        c1 = (inputShape_C + block_size - 1) // block_size
        file_6d_shape = (c1, 1, ksize_h, ksize_w, block_size, block_size)

        file_5d = _NCHW_to_NC1C0HW(depthwise_grad_kernel_KCHW).transpose(1, 3, 4, 0, 2)
        kernel_matrix_value = np.zeros(file_6d_shape, dtype=np.float16)
        for d4 in range(block_size):
            for d5 in range(block_size):
                if d4 == d5:
                    kernel_matrix_value[:, 0, :, :, d4, d5] = file_5d[:, :, :,0, d5]
        # input_grad_5d
        x1 = np.random.randn(*inputShape_NCHW).astype(np.float32)
        pool_out_no_mean_NCHW = conv_forward_naive(x1, depthwise_grad_kernel_KCHW, strides, padding_mode,
                                                   pads, ceil_mode, exclusive)
        mean_matrix_value = mean_matrix_value.astype(np.float16)
        kernel_matrix_value = kernel_matrix_value.astype(np.float16)
    pool_out_NCHW = pool_out_no_mean_NCHW
    pool_out_NC1HWC0 = _NCHW_to_NC1HWC0(pool_out_NCHW)
    pool_out_NC1HWC0 = pool_out_NC1HWC0.astype(np.float16)

    return mean_matrix_value, kernel_matrix_value, pool_out_NC1HWC0

def gen_output_value(orig_input_shape, ksize, strides, data_format, padding_mode,
                     global_pooling, mean_matrix_value, input_grad_value, pads, ceil_mode):
    ####
    inputShape = orig_input_shape
    if data_format == "NCHW":
        inputShape_N, inputShape_C, inputShape_H, inputShape_W = inputShape
        _, _, ksize_h, ksize_w = ksize
        _, _, stride_h, stride_w = strides
    elif data_format == "NHWC":
        inputShape_N, inputShape_H, inputShape_W, inputShape_C = inputShape
        _, ksize_h, ksize_w, _ = ksize
        _, stride_h, stride_w, _ = strides
    else:
        raise RuntimeError("only support NCHW NHWC,")

    if not global_pooling:
        strides = 1, stride_h, stride_w, 1
        inputShape_NHWC = inputShape_N, inputShape_H, inputShape_W, inputShape_C

        pool_out_NCHW = _NC1HWC0_to_NCHW(input_grad_value)
        mean_value_table_NCHW = _NC1HWC0_to_NCHW(mean_matrix_value)
        mean_value_table_NCHW = mean_value_table_NCHW.astype(np.float32)
        pool_out_NCHW = pool_out_NCHW.astype(np.float32)
        depthwise_grad_kernel_shape = (1, inputShape_C, ksize_h, ksize_w)
        depthwise_grad_kernel_KCHW = np.ones(depthwise_grad_kernel_shape)
        depthwise_grad_kernel_HWCK = depthwise_grad_kernel_KCHW.transpose(2,3,1,0)

        mean_value_pool_out_NCHW = np.multiply(mean_value_table_NCHW, pool_out_NCHW)
        mean_value_pool_out_NCHW = mean_value_pool_out_NCHW.astype(np.float16)
        mean_value_pool_out_NCHW = mean_value_pool_out_NCHW.astype(np.float32)
        mean_value_pool_out_NHWC = mean_value_pool_out_NCHW.transpose(0,2,3,1)
        simulator_depthwise_grad_input_NHWC = depthwise_grad(inputShape_NHWC, depthwise_grad_kernel_HWCK,
                                                             mean_value_pool_out_NHWC, strides, padding_mode,
                                                             pads, ceil_mode)
        simulator_depthwise_backward_input_NCHW = simulator_depthwise_grad_input_NHWC.transpose(0,3,1,2)
        output_grad_5d = _NCHW_to_NC1HWC0(simulator_depthwise_backward_input_NCHW)
    else:
        output_grad_5d = input_grad_value / (inputShape_H * inputShape_W)
    output_grad_5d = output_grad_5d.astype(np.float16)

    return output_grad_5d

# [TODO] coding expect function here
def avgpoolgrad_expect_func(input_grad,
                            mean_matrix,
                            #kernel_matrix,
                            out_grad,
                            orig_input_shape,
                            ksize,
                            strides,
                            padding_mode="CALCULATED",
                            pads=(0,0,0,0),
                            data_format='NCHW',
                            global_pooling=False,
                            ceil_mode=False
                            #exclusive=True,
                            #kernel_name="avg_pool_v2_grad"
                            ):
    input_grad_value = input_grad.get('value')
    if mean_matrix is not None:
        mean_matrix_value = mean_matrix.get('value')
    else:
        mean_matrix_value = None
    output_grad_value = gen_output_value(orig_input_shape, ksize, strides, data_format, padding_mode, global_pooling,
                                         mean_matrix_value, input_grad_value, pads, ceil_mode)
    if global_pooling:
        output_grad_value = np.broadcast_to(output_grad_value, out_grad.get('shape'))

    return output_grad_value
########
case1 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (2, 16, 1, 1),
                "shape": (2, 1, 1, 1, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 1, 1),
                "shape": (1, 1, 1, 1, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "CALCULATED", (1, 1, 1, 1),
               "NCHW", False, False, True],
    "case_name": "avg_pool_v2_grad_d_wrong_dim_n",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case2 = {
        "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                    "shape": (1, 1, 8, 8, 16), "param_type": "input",},
                   {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                    "shape": (1, 1, 8, 8, 16), "param_type": "input",},
                   {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                    "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
                   {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                    "shape": (1, 1, 15, 15, 16), "param_type": "output"},
                   [1, 32, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "CALCULATED", (1, 1, 1, 1),
                   "NCHW", False, False, True],
        "case_name": "avg_pool_v2_grad_d_wrong_outshape",
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True
}
case3 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 32, 15, 15),
                "shape": (1, 2, 15, 15, 16), "param_type": "output"},
               [1, 32, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "CALCULATED", (1, 1, 1, 1),
               "NCHW", False, False, True],
    "case_name": "avg_pool_v2_grad_d_wrong_dim_c",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case4 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "CALCULATED", (1, 1, 1, 1),
               "NCHW", True, False, True],
    "case_name": "avg_pool_v2_grad_d_global_wrong_shape",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case5 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 2, 2], [1, 1, 0, 0], "CALCULATED", (1, 1, 1, 1),
               "NCHW", True, False, True],
    "case_name": "avg_pool_v2_grad_d_wrong_stride",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case6 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 1, 256], [1, 1, 1, 1], "CALCULATED", (1, 1, 1, 1),
               "NCHW", False, False, True],
    "case_name": "avg_pool_v2_grad_d_global_wrong_ksize",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case7 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "padding_mode", (1, 1, 1, 1),
               "NCHW", False, False, True],
    "case_name": "avg_pool_v2_grad_d_global_wrong_padding_mode",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case8 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "CALCULATED", (10, 10, 1, 1),
               "NCHW", False, False, True],
    "case_name": "avg_pool_v2_grad_d_global_wrong_pads",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case9 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 8, 8),
                "shape": (1, 1, 8, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "CALCULATED", (1, 1, 1, 1),
               "NCHW", False, True, True],
    "case_name": "avg_pool_v2_grad_d_global_wrong_out_shape_h",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case10 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 9, 8),
                "shape": (1, 1, 9, 8, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 9, 9),
                "shape": (1, 1, 9, 9, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "CALCULATED", (1, 1, 1, 1),
               "NCHW", False, True, True],
    "case_name": "avg_pool_v2_grad_d_global_wrong_out_shape_w",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case11 = {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 9, 9),
                "shape": (1, 1, 9, 9, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 9, 9),
                "shape": (1, 1, 9, 9, 16), "param_type": "input",},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "HWCN", "ori_shape": (2, 2, 16, 1),
                "shape": (1, 2, 2, 1, 16, 16), "param_type": "input",},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 16, 15, 15),
                "shape": (1, 1, 15, 15, 16), "param_type": "output"},
               [1, 16, 15, 15], [1, 1, 2, 2], [1, 1, 2, 2], "CALCULATED", (1, 1, 1, 1),
               "NCHW", False, True, True],
    "case_name": "avg_pool_v2_grad_d_global_wrong_out_shape_w",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
ut_case.add_case(["Ascend310", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend910A"], case11)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
