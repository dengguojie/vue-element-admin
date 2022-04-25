#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for pooling pattern
"""
# Third-Party Packages
import numpy as np
import tbetoolkits
from .convolutional_utils import due_overflow
from .convolutional_utils import _getPadList
from ..registry import register_golden


# noinspection PyUnusedLocal
@register_golden(["avg_pool"])
def _avgpool(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x, conv_filter, assist_matrix, bias = context.input_arrays
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    padding = context.other_runtime_params.get("padding")
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    output_dtype = context.output_dtypes
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("avgpool testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    print("[WARNING] avgpool_mul golden func")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    ksize_h, ksize_w = ksize[h_index], ksize[w_index]
    IN, IC, IH, IW, C0 = x.shape
    ON = IN
    OC = IC * C0
    if padding == "VALID":
        OH = (IH - ksize_h) // strideh + 1
        OW = (IW - ksize_w) // stridew + 1
    else:
        OH = (IH + strideh - 1) // strideh
        OW = (IW + stridew - 1) // stridew
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    avg_pool_result = tf.compat.v1.nn.avg_pool(tensor_x, ksize=[1, ksize_h, ksize_w, 1],
                                               strides=[1, strideh, stridew, 1],
                                               padding=padding, data_format="NHWC")
    feed_dict = {tensor_x: x}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(avg_pool_result, feed_dict=feed_dict)

    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)
                         ).transpose(0, 3, 1, 2, 4)
    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output


# noinspection PyUnusedLocal
@register_golden(["avg_pool_grad"])
def _avgpoolgrad(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    input_grad, mean_matrix, kernel_matrix = context.input_arrays
    orig_input_shape = context.other_runtime_params.get("orig_input_shape")
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    padding = context.other_runtime_params.get("padding")
    data_format = context.other_runtime_params.get("data_format", "NHWC")
    output_dtype = context.output_dtypes

    # input_grad = [n, c1, 1, dy_h, dy_w, c0]
    # Collect shape info
    print("[WARNING] AvgpoolGrad golden func")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    ksize_h, ksize_w = ksize[h_index], ksize[w_index]
    IN, IC, IH, IW, C0 = input_grad.shape
    if data_format == 'NCHW':
        N, C, H, W = orig_input_shape
    else:
        N, H, W, C = orig_input_shape
    input_grad = input_grad.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    C = (C + 15) // 16 * 16
    orig_input_shape = (N, H, W, C)
    tensor_dy = tf.compat.v1.placeholder(shape=input_grad.shape, dtype=np.float32)
    avgpoolgrad_result = tf.compat.v1.raw_ops.AvgPoolGrad(orig_input_shape=orig_input_shape, grad=tensor_dy,
                                                          ksize=[1, ksize_h, ksize_w, 1],
                                                          strides=[1, strideh, stridew, 1],
                                                          padding=padding, data_format="NHWC",
                                                          name="avg_pool_grad")
    feed_dict = {tensor_dy: input_grad}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(avgpoolgrad_result, feed_dict=feed_dict)
    # NHWC to NC1HWC0
    output = out.reshape((N, H, W, C // 16, 16)).transpose(0, 3, 1, 2, 4)
    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output


# noinspection PyUnusedLocal
@register_golden(["avg_pool_v2"])
def _avgpool_v2_np(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x, conv_filter, assist_matrix = context.input_arrays
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    padding = context.other_runtime_params.get("padding", "CALCULATED")
    pads = context.other_runtime_params.get("pads", (0, 0, 0, 0))
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    global_pooling = context.other_runtime_params.get("global_pooling", False)
    ceil_mode = context.other_runtime_params.get("ceil_mode", False)
    exclusive = context.other_runtime_params.get("exclusive", True)
    output_dtype = context.output_dtypes
    # 5HD input only
    print("[WARNING] dynamic_avgpool golden func")
    if len(x.shape) != 5:
        raise RuntimeError("avgpool testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    ksize_h, ksize_w = ksize[h_index], ksize[w_index]
    IN, IC, IH, IW, C0 = x.shape
    ON = IN
    OC = IC * C0
    # x filter to NCHW
    x = x.transpose(0, 1, 4, 2, 3).reshape(IN, IC * C0, IH, IW).astype(np.float32)
    # 5HD to HWCN
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)

    # resize window
    if ksize_h > IH:
        ksize_h = IH
    if ksize_w > IW:
        ksize_w = IW
    if global_pooling or (ksize_h >= IH and ksize_w >= IW):
        ksize_h = IH
        ksize_w = IW
        padding = "VALID"
    pad, dy_shape = _getPadList(padding, [IN, IC * C0, IH, IW], [IC * C0, 16, ksize_h, ksize_w], None,
                                [strideh, stridew], pads=pads, ceil_mode=ceil_mode)
    padt, padb, padl, padr = pad
    _, _, OH, OW = dy_shape

    res_out_NCHW = np.zeros(dy_shape).astype(np.float32)
    if padding == "VALID":
        valid_h = (OH - 1) * strideh + ksize_h
        valid_w = (OW - 1) * stridew + ksize_w
        valid_shape = (IN, IC * C0, valid_h, valid_w)
        tensor_in_with_pad = np.zeros(valid_shape).astype(np.float16)
        tensor_in_with_pad[:, :, : valid_h, : valid_w] = x[:, :, : valid_h, : valid_w]
        for i in range(OH):
            for j in range(OW):
                tensor_in_mask_with_window = tensor_in_with_pad[:, :, i * strideh: i * strideh + ksize_h,
                                             j * stridew: j * stridew + ksize_w]
                res_out_NCHW[:, :, i, j] = 1.0 * np.sum(tensor_in_mask_with_window, axis=(2, 3)) / (ksize_h * ksize_w)
    else:
        same_shape = (IN, IC * C0, IH + padt + padb, IW + padl + padr)
        tensor_in_with_pad = np.zeros(same_shape).astype(np.float16)
        tensor_in_with_pad[:, :, padt: IH + padt, padl: IW + padl] = x
        # compute avg sum
        for i in range(OH):
            for j in range(OW):
                tensor_in_mask_with_window = tensor_in_with_pad[:, :, i * strideh: i * strideh + ksize_h,
                                             j * stridew: j * stridew + ksize_w]
                res_out_NCHW[:, :, i, j] = np.sum(tensor_in_mask_with_window, axis=(2, 3))
        # compute mean factor
        avg_mean_factor = []
        for i in range(OH):
            for j in range(OW):
                h_start = i * strideh - padt
                w_start = j * stridew - padl
                h_end = min(h_start + ksize_h, IH)
                w_end = min(w_start + ksize_w, IW)
                h_start = max(h_start, 0)
                w_start = max(w_start, 0)

                area = max((h_end - h_start) * (w_end - w_start), 1)
                mean_value = 1.0 / float(area)
                avg_mean_factor.append(mean_value)
        avg_mean_factor = np.array(avg_mean_factor).reshape(OH, OW)
        # compute res out
        for i in range(OH):
            for j in range(OW):
                res_out_NCHW[:, :, i, j] = res_out_NCHW[:, :, i, j] * avg_mean_factor[i, j]

    # NCHW to NC1HWC0
    output = res_out_NCHW.reshape((IN, IC, C0, OH, OW)).transpose((0, 1, 3, 4, 2))
    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output


# noinspection PyUnusedLocal
@register_golden(["avg_pool3d_grad"])
def _avg_pool3d_grad(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    grads, _filter, multiple = context.input_arrays
    orig_input_shape = context.other_runtime_params.get("orig_input_shape")
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    ceil_mode = context.other_runtime_params.get("ceil_mode", False)
    count_include_pad = context.other_runtime_params.get("count_include_pad", True)
    divisor_override = context.other_runtime_params.get("divisor_override", 0)
    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    print("============ GET INTO AVGPOOL3DGRAD GOLDEN ===========")
    if len(grads.shape) != 6:
        raise RuntimeError("avgpool3dgrad testcase golden function supports NDC1HWC0 input only!")
    print("grads shape:", grads.shape)
    print("ksize:", ksize)
    print("orig_input_shape:", orig_input_shape)
    # Collect shape info
    n_index = data_format.index("N")
    d_index = data_format.index("D")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    c_index = data_format.index("C")
    stride_h, stride_w, stride_d = strides[h_index], strides[w_index], strides[d_index]
    filter_h, filter_w, filter_d = ksize[h_index], ksize[w_index], ksize[d_index]
    GN, GD, GC, GH, GW, C0 = grads.shape
    # IN, ID, IH, IW, IC = orig_input_shape
    IN, ID, IH, IW, IC = (orig_input_shape[n_index], orig_input_shape[d_index], orig_input_shape[h_index],
                          orig_input_shape[w_index], orig_input_shape[c_index])
    IC = (IC + 15) // 16 * 16
    if all(i == 0 for i in pads):
        padding = "VALID"
    else:
        padding = "SAME"

    # grads to NDHWC
    output_backprop = grads.transpose(0, 1, 3, 4, 2, 5).reshape(GN, GD, GH, GW, GC * C0)
    grads_tensor = tf.compat.v1.placeholder(grads.dtype, shape=output_backprop.shape)
    grads_tensor = tf.compat.v1.cast(grads_tensor, tf.float32)
    res = tf.compat.v1.raw_ops.AvgPool3DGrad(orig_input_shape=[GN, ID, IH, IW, GC * C0],
                                             grad=grads_tensor,
                                             ksize=[1, filter_d, filter_h, filter_w, 1],
                                             strides=[1, stride_d, stride_h, stride_w, 1],
                                             padding=padding,
                                             data_format="NDHWC",
                                             name='avg_pool3d_grad')
    res = tf.compat.v1.cast(res, tf.float16)
    feed_dict = {grads_tensor: output_backprop}
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(res, feed_dict=feed_dict)

    # import torch
    # from torch.autograd import Variable as V
    # output_backprop = grads.transpose(0,2,5,1,3,4).reshape(GN, GC * C0, GD, GH, GW).astype(np.float32)
    # input = V(torch.from_numpy(np.ones((IN, IC, ID, IH, IW)).astype(np.float32)), requires_grad=True)
    # op = torch.nn.AvgPool3d([filter_d, filter_h, filter_w],
    #                         [stride_d, stride_h, stride_w], [pads[0], pads[2], pads[4]], count_include_pad=False)
    # output = op(input).backward(gradient=torch.from_numpy(output_backprop), retain_graph=True)
    # res = input.grad.numpy().reshape((IN, IC//C0, C0, ID, IH, IW)).transpose(0, 3, 1, 4, 5, 2)

    # NDHWC to NDC1HWC0
    # out = np.zeros((IN, ID, IH, IW, IC), dtype=np.float16)
    res = out.reshape((IN, ID, IH, IW, IC // C0, C0)).transpose(0, 1, 4, 2, 3, 5).copy().astype(np.float16)
    # print("GOLEDN output:", res)
    return res


# noinspection PyUnusedLocal
@register_golden(["avg_pool3d"])
def _avg_pool3d(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x, _filter, multiplier = context.input_arrays
    ksize = context.other_runtime_params.get("ksize")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    ceil_mode = context.other_runtime_params.get("ceil_mode", False)
    count_include_pad = context.other_runtime_params.get("count_include_pad", True)
    divisor_override = context.other_runtime_params.get("divisor_override", 0)
    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    print("============ GET INTO AVGPOOL3D GOLDEN ===========")
    if len(x.shape) != 6:
        raise RuntimeError("avgpool3dgrad testcase golden function supports NDC1HWC0 input only!")
    n_index = data_format.index("N")
    d_index = data_format.index("D")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    c_index = data_format.index("C")

    FN, FD, FC1, FH, FW, C0 = x.shape
    stride_d, stride_h, stride_w = strides[d_index], strides[h_index], strides[w_index]
    ksize_d, ksize_h, ksize_w = ksize[d_index], ksize[h_index], ksize[w_index]
    fmap_data = x.transpose(0, 1, 3, 4, 2, 5).reshape(FN, FD, FH, FW, FC1 * C0)
    fmap_tensor = tf.compat.v1.placeholder(x.dtype, shape=[FN, FD, FH, FW, FC1 * C0])
    fmap_tensor = tf.compat.v1.cast(fmap_tensor, tf.float32)
    padding = "VALID" if all(i == 0 for i in pads) else "SAME"
    res = tf.compat.v1.nn.avg_pool3d(fmap_tensor, [1, ksize[d_index], ksize[h_index], ksize[w_index], 1],
                                     [1, strides[d_index], strides[h_index], strides[w_index], 1],
                                     padding, data_format="NDHWC")
    res = tf.compat.v1.cast(res, tf.float16)
    feed_dict = {fmap_tensor: fmap_data}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        out = sess.run(res, feed_dict=feed_dict)
    # print("out:", out)

    # OD = (FD + pads[0] + pads[1] - ksize_d) // stride_d + 1
    # OH = (FH + pads[2] + pads[3] - ksize_h) // stride_h + 1
    # OW = (FW + pads[4] + pads[5] - ksize_w) // stride_w + 1
    # out = np.zeros((FN, OD, OH, OW, FC1 * C0), dtype=np.float16)
    # print("output shape:", (FN, OD, OH, OW, FC1 * C0))

    ON, OD, OH, OW, OC = out.shape
    res = out.reshape((FN, OD, OH, OW, FC1, C0)).transpose(0, 1, 4, 2, 3, 5).copy().astype(np.float16)
    print("create golden finish.")
    return res
