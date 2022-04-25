#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for convolution back-propagation pattern
"""
# Third-Party Packages
import numpy as np

import tbetoolkits
from .convolutional_utils import _calculate_group
from .convolutional_utils import _ceil
from .convolutional_utils import _conv2d_bp_np
from .convolutional_utils import _conv2d_dx_golden
from .convolutional_utils import _conv_bp_filter
from .convolutional_utils import _gen_depthwise_conv2d_backprop_filter_data
from .convolutional_utils import _getPadding
from .convolutional_utils import _native_fun
# noinspection PyUnresolvedReferences
from .convolutional_utils import due_overflow
from ..registry import register_golden


@register_golden(["conv2d_backprop_input", "conv2d_bp_input_transdata"])
def _conv2d_backprop_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    _, filter_ori, out_backprop = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    input_formats = context.stc_input_formats
    output_dtype = context.output_dtypes
    output_formats = context.output_formats[0]
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]

    if data_format == 'NCHW':
        Ni, cin_ori, Hi, Wi = input_size
    else:
        Ni, Hi, Wi, cin_ori = input_size
    w_ori_format = ori_formats[0]
    w_ori_shape = ori_shapes[0]
    cout_ori = w_ori_shape[w_ori_format.index("N")]
    out_backprop_formats = input_formats[1]
    output = _conv2d_dx_golden(out_backprop, filter_ori, [Ni, cin_ori, Hi, Wi], cout_ori, [
        strideh, stridew], pads, [dilationh, dilationw], groups, out_backprop_formats, output_formats)

    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output


# noinspection PyUnusedLocal
@register_golden(["conv2d_transpose"])
def _conv2d_transpose(context: "tbetoolkits.UniversalTestcaseStructure"):
    _, out_backprop, filter_ori, bias, offset_w = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    output_padding = context.other_runtime_params.get("output_padding", (0, 0, 0, 0))
    offset_x = context.other_runtime_params.get("offset_x", 0)
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    # 5HD input only
    if len(out_backprop.shape) != 5:
        raise RuntimeError(
            "conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]

    if data_format == 'NCHW':
        Ni, cin_ori, Hi, Wi = input_size
    else:
        Ni, Hi, Wi, cin_ori = input_size
    w_ori_format = ori_formats[1]
    w_ori_shape = ori_shapes[1]
    cout_ori = w_ori_shape[w_ori_format.index("N")]

    # ori_dy_dtype = out_backprop.dtype
    output = _conv2d_dx_golden(out_backprop, filter_ori, [Ni, cin_ori, Hi, Wi], cout_ori, [
        strideh, stridew], pads, [dilationh, dilationw], groups, bias=bias, output_dtype=output_dtype[0])

    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output


@register_golden(["deconvolution"])
def _deconvolution(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    out_backprop, filter_ori, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    # 5HD input only
    if len(out_backprop.shape) != 5:
        raise RuntimeError(
            "conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides
    dilationh, dilationw = dilations[h_index], dilations[w_index]

    batch, cout1, Ho, Wo, block_size = out_backprop.shape
    k_c1, hk, wk, cout_g, _ = filter_ori.shape
    w_ori_format = ori_formats[1]
    w_ori_shape = ori_shapes[1]
    cout_ori = w_ori_shape[w_ori_format.index("N")]
    cin_ori = w_ori_shape[w_ori_format.index("C")] * groups

    he = (hk - 1) * dilationh + 1
    we = (wk - 1) * dilationw + 1
    Hi = strideh * (Ho - 1) + he - pad_top - pad_bottom
    Wi = stridew * (Wo - 1) + we - pad_left - pad_right

    # ori_dy_dtype = out_backprop.dtype
    # output = conv2d_dx_golden(out_backprop, filter_ori, [batch, cin_ori, Hi, Wi], cout_ori, strides, pads,
    #                           [dilationh, dilationw], groups, bias)

    # pad_top, pad_bottom, pad_left, pad_right = pads
    # strideh, stridew = strides
    # dilationh, dilationw = dilations

    # batch, cout1, Ho, Wo, block_size = out_backprop.shape
    # k_c1, hk, wk, cout_g, _ = filter_ori.shape
    # Ni, cin_ori, Hi, Wi = input_size
    Ni = batch
    cin1 = _ceil(cin_ori, block_size)
    cin = cin1 * block_size
    cout = cout1 * block_size

    ori_dy_dtype = out_backprop.dtype
    if ori_dy_dtype == 'float16':
        filter_ori = filter_ori.astype('float32')
        out_backprop = out_backprop.astype('float32')
    padding, _ = _getPadding(pads, [Ni, Hi, Wi, cin], [hk, wk, k_c1 * block_size, cout_g], [batch, Ho, Wo, cout],
                             (strideh, stridew), [dilationh, dilationw])
    if groups == 1:
        # 5HD to NHWC
        dy = out_backprop.transpose(0, 2, 3, 1, 4).reshape(
            batch, Ho, Wo, cout)
        # C1HWNC0 to HWCN
        w = filter_ori.transpose(1, 2, 0, 4, 3).reshape(
            hk, wk, k_c1 * block_size, cout_g)

        tensor_filter = tf.compat.v1.placeholder(w.dtype, shape=w.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        tf_dx = tf.compat.v1.nn.conv2d_backprop_input([Ni, Hi, Wi, cin],
                                                      tensor_filter,
                                                      tensor_dy,
                                                      strides=[1, strideh, stridew, 1],
                                                      padding=(
                                                        (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                                      data_format='NHWC',
                                                      use_cudnn_on_gpu=False,
                                                      dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_filter: w, tensor_dy: dy}
        if bias is not None:
            tf_dx = tf.compat.v1.cast(tf_dx, bias.dtype)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            tf_dx = tf.compat.v1.nn.bias_add(tf_dx, tensor_bias)
            feed_dict[tensor_bias] = bias
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(tf_dx, feed_dict=feed_dict)

        # NHWC to NC1HWC0
        output = out.reshape((Ni, Hi, Wi, cin1, block_size)).transpose(
            0, 3, 1, 2, 4)
    elif groups == cin_ori and strideh == stridew and padding in ('SAME', 'VALID'):
        # 5HD to NHWC
        dy = out_backprop.transpose(0, 2, 3, 1, 4).reshape(batch, Ho, Wo, cout)
        # C1HWNC0 _> NCHW--> HWCN HWC1
        multi = cout_ori // cin_ori
        w = _conv_bp_filter(filter_ori, multi, block_size).transpose(2, 3, 1, 0)

        # tf.compat.v1.enable_eager_execution()
        tensor_filter = tf.compat.v1.placeholder(w.dtype, shape=w.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        dp_dx = tf.compat.v1.nn.depthwise_conv2d_backprop_input([Ni, Hi, Wi, cin], tensor_filter, tensor_dy,
                                                                strides=[1, strideh, stridew, 1],
                                                                padding=padding, data_format="NHWC",
                                                                dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_dy: dy, tensor_filter: w}
        if bias is not None:
            dp_dx = tf.compat.v1.cast(dp_dx, bias.dtype)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            dp_dx = tf.compat.v1.nn.bias_add(dp_dx, tensor_bias)
            feed_dict[tensor_bias] = bias
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(dp_dx, feed_dict=feed_dict)

        # N, H, W, Ci_ori -> N, H, W, Ci -> N, Ci1, H, W, Ci0
        # out = np.pad(out, ((0, 0), (0, 0), (0, 0), (0, cin - cin_ori)), 'constant')
        # NHWC to NC1HWC0
        output = out.reshape((Ni, Hi, Wi, cin1, block_size)
                             ).transpose(0, 3, 1, 2, 4)
    # elif pad_top == pad_bottom and pad_left == pad_right:
    #     import torch
    #     torch.backends.cudnn.benchmark = True

    #     group_dict = _calculate_group(cin_ori, cout_ori, groups)
    #     ### 5HD to NCHW
    #     dy = out_backprop.transpose(0, 1, 4, 2, 3).reshape(batch, cout, Ho, Wo)
    #     ### dy_nhwc = out_backprop.transpose(0, 2, 3, 1, 4).reshape(batch, Ho, Wo, cout)
    #     ### dy = dy_nhwc[:, :, :, :cout_ori].transpose(0, 3, 1, 2)
    #     ### C1HWNC0 to NCHW
    #     multi = cout_ori // cin_ori
    #     w = conv_bp_filter(filter_ori, multi, block_size).reshape(cout, 1, hk, wk)
    #     ### w = conv_group_filter(filter_ori, group_dict)
    #     if bias is not None:
    #         print("bias_shape", bias.shape)
    #         pytorch_bias = torch.from_numpy(bias.astype(output_dtype[0]))
    #     else:
    #         pytorch_bias = None
    #     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx', out_backprop.shape, dy.shape,  filter_ori.shape, w.shape)
    #     pytorch_dy = torch.from_numpy(dy) # .type(torch.float32)
    #     pytorch_w = torch.from_numpy(w)
    #     print('xxxxxxxxxxxtorch_dx:', [Ni, cin, Hi, Wi],
    #         filter_ori.shape, pytorch_w.shape, pytorch_dy.shape, groups)
    #     out = torch.nn.functional.conv_transpose2d(
    #         pytorch_dy, pytorch_w, bias=pytorch_bias, stride=[strideh, stridew],
    #         padding=[pad_top, pad_left], output_padding=0, groups=cin,
    #         dilation=[dilationh, dilationw])
    #     out = out.numpy()
    #     print('xxxxxxxxxxxtorch_dx:',  out.shape, [Ni, cin, Hi, Wi])
    #     ###out = np.pad(out, ((0, 0), (0, cin - cin_ori), (0, 0), (0, 0)), 'constant')
    #     ### NCHW to NC1HWC0
    #     output = out.reshape((Ni, cin1, block_size, Hi, Wi)).transpose(0, 1, 3, 4, 2)
    else:
        group_dict = _calculate_group(cin_ori, cout_ori, groups)
        # C1HWNC0 --> NCHW
        filter_dilate = filter_ori.transpose(3, 0, 4, 1, 2).reshape(
            cout_g, k_c1 * block_size, hk, wk)
        output = _conv2d_bp_np(filter_dilate, out_backprop, [Ni, cin, Hi, Wi], [
            strideh, stridew], pads, group_dict)
        if bias is not None:
            output = output.astype(bias.dtype)
            output = output + bias[np.newaxis, :, np.newaxis, np.newaxis]
        output = output.reshape((Ni, cin1, block_size, Hi, Wi)
                                ).transpose(0, 1, 3, 4, 2)

    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output


# noinspection PyUnusedLocal
# def _deconvolution_np(context: "tbetoolkits.UniversalTestcaseStructure"):
#     out_backprop, filter_ori, bias, offset_w = context.input_arrays
#     strides = context.other_runtime_params.get("strides")
#     pads = context.other_runtime_params.get("pads")
#     dilations = context.other_runtime_params.get("dilations")
#     groups = context.other_runtime_params.get('groups', 1)
#     data_format = context.other_runtime_params.get("data_format", "NCHW")
#     offset_x = context.other_runtime_params.get("offset_x", 0)
#     ori_shapes = context.stc_ori_inputs
#     ori_formats = context.stc_input_ori_formats
#     output_dtype = context.output_dtypes
#     # 5HD input only
#     if len(out_backprop.shape) != 5:
#         raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
#     # Collect shape info
#     h_index = data_format.index("H")
#     w_index = data_format.index("W")
#     pad_top, pad_bottom, pad_left, pad_right = pads
#     strideh, stridew = strides
#     dilationh, dilationw = dilations[h_index], dilations[w_index]
#     IN, IC, IH, IW, C0 = out_backprop.shape
#     WC, WH, WW, WN, _ = conv_filter.shape
#
#     # filter to NCHW
#     out_backprop = out_backprop.transpose(0, 1, 4, 2, 3).reshape(IN, IC * C0, IH, IW).astype(np.float32)
#     N, C = IN, WC * C0
#     he = (WH - 1) * dilationh + 1
#     we = (WW - 1) * dilationw + 1
#     H = strideh * (IH - 1) + he - pad_top - pad_bottom
#     W = stridew * (IW - 1) + we - pad_left - pad_right
#     x_shape = (N, C, H, W)
#     # 5HD to NCHW
#     conv_filter = conv_filter.transpose(3, 0, 4, 1, 2).reshape(WN, WC * C0, WH, WW).astype(np.float32)
#
#     F = IC * C0
#     dx = np.zeros(x_shape)
#     dx_pad = np.pad(dx, [(0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)])
#     for n in range(IN):
#         for i in range(IH):
#             for j in range(IW):
#                 for f in range(F):
#                     dx_pad[n, :, i * strideh:i * strideh + WH, j * stridew:j * stridew + WW] += conv_filter[f] * \
#                                                                                                 out_backprop[n, f, i, j]
#     dx = dx_pad[:, :, pad_top:pad_top + H, pad_left:pad_left + W]
#
#     # NCHW to NC1HWC0
#     output = dx.reshape((N, C // 16, 16, H, W)).transpose(0, 1, 3, 4, 2)
#     return output


@register_golden(["depthwise_conv2d_backprop_input"])
def _depthwise_conv2d_backprop_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    _, filter_ori, out_backprop = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    print('xxxxxxxxxxxxxxxxxxxxxxxdp_dx:', input_size, filter_ori.shape, out_backprop.shape)
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]

    if data_format == 'NCHW':
        Ni, cin_ori, Hi, Wi = input_size
    else:
        Ni, Hi, Wi, cin_ori = input_size
    w_ori_format = ori_formats[0]
    w_ori_shape = ori_shapes[0]
    multi = w_ori_shape[w_ori_format.index("C")]

    output = _conv2d_dx_golden(out_backprop, filter_ori, [Ni, cin_ori, Hi, Wi], cin_ori * multi, [
        strideh, stridew], pads, [dilationh, dilationw], groups=cin_ori)

    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output


@register_golden(["conv2d_backprop_filter"])
def _conv2d_backprop_filter(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    import torch
    # noinspection PyUnresolvedReferences
    from torch.autograd import Variable
    x, _, out_backprop = context.input_arrays
    filter_size = context.other_runtime_params.get("filter_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    output_ori_formats = context.output_ori_formats
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    YN, YC, YH, YW, C0 = out_backprop.shape
    if output_ori_formats[0] == 'NCHW':
        Co, C, kh, kw = filter_size
    elif output_ori_formats[0] == 'NHWC':
        Co, kh, kw, C = filter_size
    else:
        kh, kw, C, Co = filter_size
    Co1 = _ceil(Co, C0)
    c_dim = ori_formats[0].index("C")
    cin_ori = ori_shapes[0][c_dim]
    if groups == 1 and (dilationh, dilationw) == (1, 1):
        # x filter to NHWC
        w_shape = (kh, kw, IC * C0, Co1 * C0)  # HWCN
        x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
        out_backprop = out_backprop.transpose(0, 2, 3, 1, 4).reshape(YN, YH, YW, YC * C0).astype(np.float32)
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_dy = tf.compat.v1.placeholder(out_backprop.dtype, shape=out_backprop.shape)
        tf_dw_result = tf.compat.v1.nn.conv2d_backprop_filter(tensor_x, w_shape, tensor_dy,
                                                              strides=[1, strideh, stridew, 1],
                                                              padding=(
                                                                  (0, 0), (pad_top, pad_bottom), (pad_left, pad_right),
                                                                  (0, 0)),
                                                              data_format="NHWC", use_cudnn_on_gpu=False,
                                                              dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_x: x, tensor_dy: out_backprop}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(tf_dw_result, feed_dict=feed_dict)
        # HWCN to C1HWNC0
        # output = out.transpose(2, 0, 1, 3).reshape((IC, C0, kh, kw, Co1 * C0)).transpose(0, 2, 3, 4, 1)
        # HWCN to fractal_z(C1*H*W, N1, N0, C0)
        output = out.reshape((kh, kw, IC, C0, Co1, C0)).transpose(2, 0, 1, 4, 5, 3).reshape(
            IC * kh * kw, Co1, C0, C0
        )
        return output
    # elif groups == cin_ori and strideh == stridew:
    #     padding = "SAME" if sum(pads) != 0 else "VALID"
    #     # x filter to NHWC
    #     w_shape = (kh, kw, Co1 * C0, C)  # HWCN
    #     x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    #     out_backprop = out_backprop.transpose(0, 2, 3, 1, 4).reshape(YN, YH, YW, YC * C0).astype(np.float32)
    #     tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    #     tensor_dy = tf.compat.v1.placeholder(out_backprop.dtype, shape=out_backprop.shape)
    #     dw = tf.compat.v1.nn.depthwise_conv2d_backprop_filter(tensor_x, w_shape, tensor_dy, strides=[1, strideh, stridew, 1],
    #                                                padding=padding, data_format="NHWC",
    #                                                dilations=[1, dilationh, dilationw, 1])
    #     feed_dict = {tensor_x: x, tensor_dy: out_backprop}
    #     init_op = tf.compat.v1.global_variables_initializer()
    #     with tf.compat.v1.Session() as sess:
    #         sess.run(init_op)
    #         # Generate output tf data
    #         out = sess.run(dw, feed_dict=feed_dict)
    #     # H, W, Cori, 1 -> H, W, Cori, 16 -> C1HWNC0
    #     dw = np.pad(out, ((0, 0), (0, 0), (0, 0), (0, C * C0 - C)), 'constant')
    #     print("&&&&&&&", dw.shape)
    #     # HWCN to C1HWNC0   HW,48,16 --> 1,HW,48,16
    #     output = dw.transpose(3, 0, 1, 2).reshape((C, C0, kh, kw, Co1 * C0)).transpose(0, 2, 3, 4, 1)
    #     print(output.shape)
    elif groups == cin_ori:
        group_dict = _calculate_group(cin_ori, Co, groups)
        real_g = group_dict["real_g"]
        cin_g = group_dict["cin_g"]
        cin1_g = group_dict["cin1_g"]
        cout_g = group_dict["cout_g"]
        print(group_dict)
        # dedy  (N,Co1,Ho,Wo,C0)--->(real_g, n, cout_g, ho, wo)
        # dedy_target = out_backprop.transpose(1,0,5,2,3)
        # dedy  (N,Co1,Ho,Wo,C0)--->(n, ho, wo, co)
        dedy_np_data = out_backprop.transpose(0, 2, 3, 1, 4).reshape(YN, YH, YW, YC * C0).astype(np.float32)
        # dedy
        dedy_target = np.zeros((real_g, YN, cout_g, YH, YW), dtype=np.float32)
        for i in range(YC * C0):
            dedy_target[i // cout_g, :, i % cout_g, :, :] = dedy_np_data[:, :, :, i].astype(np.float32)

        # dedx (N,Ci1,Hi,Wi,C0)--->(real_g, n, cin_g, ho, wo)
        dx_np_data = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
        dedx_target = np.zeros((real_g, IN, cin_g, IH + pad_top + pad_bottom, IW + pad_left + pad_right),
                               dtype=np.float32)
        for i in range(IC * C0):
            dedx_target[i // cin_g, :, i % cin_g, pad_top:pad_top + IH, pad_left:pad_left + IW] = dx_np_data[:, :, :,
                                                                                                  i].astype(np.float32)

        temp = np.zeros((real_g, YN, cin_g, YH, kh, YW, kw), dtype=np.float32)
        for j0 in range(YH):
            for j1 in range(kh):
                for k0 in range(YW):
                    for k1 in range(kw):
                        temp[:, :, :, j0, j1, k0,
                        k1] = dedx_target[:, :, :,
                              j0 * strideh + j1,
                              k0 * stridew + k1]

        dedx_target = temp.transpose((0, 1, 2, 4, 6, 3, 5)).reshape((real_g, YN, cin_g * kh * kw, YH, YW))

        out = np.zeros((real_g, cin_g * kh * kw, cout_g), dtype=np.float32)
        for i in range(real_g):
            for j in range(cin_g * kh * kw):
                for k in range(cout_g):
                    out[i, j, k] = np.sum(dedy_target[i, :, k, :, :] *
                                          dedx_target[i, :, j, :, :])
        print('--------tmp golden:', out.shape)
        # (real_g, cin_g * kh * kw, cout_g) to C1HWNC0
        output = out.reshape((real_g, cin1_g, C0, kh * kw, cout_g)).transpose((1, 3, 0, 4, 2)).reshape(
            cin1_g * kh * kw, real_g, cout_g, C0)
        return output
    else:
        group_dict = _calculate_group(cin_ori, Co, groups)
        real_g = group_dict["real_g"]
        cin_g = group_dict["cin_g"]
        cin1_g = group_dict["cin1_g"]
        cout_g = group_dict["cout_g"]
        mag_factor = group_dict["mag_factor"]
        fmap_c = C * groups
        padding = pads[::2]
        zero_pad_h = pad_bottom - pad_top
        zero_pad_w = pad_right - pad_left
        # torch format is NCHW
        input_data = x.transpose(0, 1, 4, 2, 3).reshape(IN, IC * C0, IH, IW).astype(np.float32)
        input_data1 = np.zeros((IN, fmap_c, IH, IW)).astype(np.float32)
        input_data1[:, :, :, :] = input_data[:, :fmap_c, :, :]
        input_pad = np.pad(input_data1, ((0, 0), (0, 0), (0, zero_pad_h), (0, zero_pad_w)), 'constant',
                           constant_values=(0, 0))
        weight_data = np.random.uniform(-1, 1, size=(Co, C, kh, kw)).astype(np.float16)
        _input = Variable(torch.from_numpy(input_pad).type(torch.float32), requires_grad=True)
        _weight = Variable(torch.from_numpy(weight_data).type(torch.float32), requires_grad=True)

        out = torch.nn.functional.conv2d(_input, _weight, stride=(strideh, stridew), padding=padding,
                                         dilation=(dilationh, dilationw), groups=groups)
        grad_data = out_backprop.transpose(0, 1, 4, 2, 3).reshape(YN, YC * C0, YH, YW).astype(np.float32)
        grad_data1 = np.zeros((YN, Co, YH, YW)).astype(np.float32)
        grad_data1[:, :, :, :] = grad_data[:, :Co, :, :]
        gradients = torch.from_numpy(grad_data1)
        out.backward(gradients, retain_graph=True)
        weight_np = _weight.grad.detach().numpy()
        # NCHW to C1HWNC0
        weight_group_data = np.ones((real_g, cin1_g, kh, kw, cout_g, 16)).astype(np.float32) * 999999
        for g in range(groups):
            for ci in range(C):
                for co in range(Co // groups):
                    try:
                        e = g % mag_factor
                        dst_cin = e * C + ci
                        dst_cout = e * (Co // groups) + co
                        src_cout = g * (Co // groups) + co
                        weight_group_data[g // mag_factor, dst_cin // 16, :, :, dst_cout, dst_cin % 16] = \
                            weight_np[src_cout, ci, :, :]
                    except:
                        # e = g % mag_factor
                        # dst_cin = e * w_c + ci
                        # dst_cout = e * (w_n // groups) + co
                        # src_cout = g * (w_n // groups) + co
                        # print("================================ Error Detected =====================================")
                        # print("weight_group shape:", weight_group.shape)
                        # print("Weight Shape : ", weight.shape)
                        # print("C0:", co)
                        # print("e : ", e)
                        # print("dst_cin :", dst_cin)
                        # print("dst_cout : ", dst_cout)
                        # print("src_cout and Ci" , src_cout, "",ci)
                        # print("mag_factor : ", mag_factor)
                        raise
        print('--------golden:', weight_group_data.shape)
        return weight_group_data


@register_golden(["conv2d_bp_filter_transdata"])
def _conv2d_bp_filter_transdata(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x, out_backprop = context.input_arrays
    print(x.shape, out_backprop.shape)
    filter_size = context.other_runtime_params.get("filter_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    # 5HD input only
    if len(x.shape) == 5:
        # Collect shape info
        h_index = data_format.index("H")
        w_index = data_format.index("W")
        pad_top, pad_bottom, pad_left, pad_right = pads
        strideh, stridew = strides[h_index], strides[w_index]
        dilationh, dilationw = dilations[h_index], dilations[w_index]
        IN, IC, IH, IW, C0 = x.shape
        YN, YC, YH, YW, C0 = out_backprop.shape
        Co, C, kh, kw = filter_size
        Co1 = _ceil(Co, C0)
        c_dim = ori_formats[0].index("C")
        cin_ori = ori_shapes[0][c_dim]
        if groups == 1:
            # x filter to NHWC
            w_shape = (kh, kw, IC * C0, Co1 * C0)  # HWCN
            x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
            out_backprop = out_backprop.transpose(0, 2, 3, 1, 4).reshape(YN, YH, YW, YC * C0).astype(np.float32)
            tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
            tensor_dy = tf.compat.v1.placeholder(out_backprop.dtype, shape=out_backprop.shape)
            tf_dw_result = tf.compat.v1.nn.conv2d_backprop_filter(tensor_x, w_shape, tensor_dy,
                                                                  strides=[1, strideh, stridew, 1],
                                                                  padding=(
                                                                      (0, 0), (pad_top, pad_bottom),
                                                                      (pad_left, pad_right), (0, 0)),
                                                                  data_format="NHWC", use_cudnn_on_gpu=False,
                                                                  dilations=[1, dilationh, dilationw, 1])
            feed_dict = {tensor_x: x, tensor_dy: out_backprop}
            init_op = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:
                sess.run(init_op)
                # Generate output tf data
                out = sess.run(tf_dw_result, feed_dict=feed_dict)
            # HWCN to C1HWNC0
            output = out.transpose(2, 0, 1, 3).reshape((IC, C0, kh, kw, Co1 * C0)).transpose(0, 2, 3, 4, 1)
        else:
            group_dict = _calculate_group(cin_ori, Co, groups)
            real_g = group_dict["real_g"]
            cin_g = group_dict["cin_g"]
            cin1_g = group_dict["cin1_g"]
            cout_g = group_dict["cout_g"]
            print(group_dict)
            # dedy  (N,Co1,Ho,Wo,C0)--->(real_g, n, cout_g, ho, wo)
            # dedy_target = out_backprop.transpose(1,0,5,2,3)
            # dedy  (N,Co1,Ho,Wo,C0)--->(n, ho, wo, co)
            dedy_np_data = out_backprop.transpose(0, 2, 3, 1, 4).reshape(YN, YH, YW, YC * C0).astype(np.float32)
            # dedy
            dedy_target = np.zeros((real_g, YN, cout_g, YH, YW), dtype=np.float32)
            for i in range(YC * C0):
                dedy_target[i // cout_g, :, i % cout_g, :, :] = dedy_np_data[:, :, :, i].astype(np.float32)

            # dedx (N,Ci1,Hi,Wi,C0)--->(real_g, n, cin_g, ho, wo)
            dx_np_data = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
            dedx_target = np.zeros((real_g, IN, cin_g, IH + pad_top + pad_bottom, IW + pad_left + pad_right),
                                   dtype=np.float32)
            for i in range(IC * C0):
                dedx_target[i // cin_g, :, i % cin_g, pad_top:pad_top + IH, pad_left:pad_left + IW] = dx_np_data[
                    :, :,:, i].astype(np.float32)

            temp = np.zeros((real_g, YN, cin_g, YH, kh, YW, kw), dtype=np.float32)
            for j0 in range(YH):
                for j1 in range(kh):
                    for k0 in range(YW):
                        for k1 in range(kw):
                            temp[:, :, :, j0, j1, k0,
                            k1] = dedx_target[:, :, :,
                                              j0 * strideh + j1,
                                              k0 * stridew + k1]

            dedx_target = temp.transpose((0, 1, 2, 4, 6, 3, 5)).reshape((real_g, YN, cin_g * kh * kw, YH, YW))

            out = np.zeros((real_g, cin_g * kh * kw, cout_g), dtype=np.float32)
            for i in range(real_g):
                for j in range(cin_g * kh * kw):
                    for k in range(cout_g):
                        out[i, j, k] = np.sum(dedy_target[i, :, k, :, :] *
                                              dedx_target[i, :, j, :, :])
            print('--------tmp golden:', out.shape)
            # (real_g, cin_g * kh * kw, cout_g) to C1HWNC0
            output = out.reshape((real_g, cin1_g, C0, kh, kw, cout_g)).transpose((0, 1, 3, 4, 5, 2)).reshape(
                real_g, cin1_g, kh, kw, cout_g, C0)
        print('--------golden:', output.shape)
        return output
    elif len(x.shape) == 4:
        if context.other_runtime_params.get("data_format", "NCHW") != "NCHW" or groups != 1:
            raise RuntimeError(
                "dw 4D format support NCHW and group=1 input only!")
        print("=============== 4d scene begin =================")
        # Collect shape info
        h_index = data_format.index("H")
        w_index = data_format.index("W")
        pad_top, pad_bottom, pad_left, pad_right = pads
        strideh, stridew = strides[h_index], strides[w_index]
        dilationh, dilationw = dilations[h_index], dilations[w_index]
        IN, IC, IH, IW = x.shape
        YN, YC, YH, YW = out_backprop.shape
        Co, C, kh, kw = filter_size
        c_dim = ori_formats[0].index("C")
        cin_ori = ori_shapes[0][c_dim]
        w_shape = (kh, kw, IC, YC)  # HWCN
        x = x.transpose(0, 2, 3, 1).astype(np.float32)
        out_backprop = out_backprop.transpose(0, 2, 3, 1).astype(np.float32)
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_dy = tf.compat.v1.placeholder(out_backprop.dtype, shape=out_backprop.shape)
        tf_dw_result = tf.compat.v1.nn.conv2d_backprop_filter(tensor_x, w_shape, tensor_dy,
                                                              strides=[1, strideh, stridew, 1],
                                                              padding=((0, 0), (pad_top, pad_bottom),
                                                                       (pad_left, pad_right), (0, 0)),
                                                              data_format="NHWC", use_cudnn_on_gpu=False,
                                                              dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_x: x, tensor_dy: out_backprop}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(tf_dw_result, feed_dict=feed_dict)
        # HWCN to C1HWNC0
        output = out.transpose(2, 0, 1, 3)
        dw_align = np.zeros((_ceil(IC, 16) * 16, kh, kw, _ceil(YC, 16) * 16), dtype=np.float32)
        for i in range(IC):
            for j in range(YC):
                dw_align[i, :, :, j] = output[i, :, :, j]
        dw_align = dw_align.reshape((_ceil(IC, 16), 16, kh, kw, _ceil(YC, 16) * 16)).transpose(0, 2, 3, 4, 1)
        return dw_align


# noinspection PyUnusedLocal
@register_golden(["depthwise_conv2d_backprop_filter"])
def _depthwise_conv2d_backprop_filter(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x, _, out_backprop = context.input_arrays
    filter_size = context.other_runtime_params.get("filter_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    output_dtype = context.output_dtypes
    filter_format = context.output_ori_formats[0]
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    # pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    # dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    YN, YC, YH, YW, C0 = out_backprop.shape
    if filter_format == 'NCHW':
        k_c, multi, kh, kw = filter_size
    else:
        kh, kw, multi, k_c = filter_size

    if strideh == stridew:
        # x filter to NHWC
        x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
        out_backprop = out_backprop.transpose(0, 2, 3, 1, 4).reshape(YN, YH, YW, YC * C0).astype(np.float32)
        # Co1 = (Co + C0 - 1) // C0
        w_shape = (kh, kw, IC * C0, 1)  # HWCN

        if list(pads) == [0, 0, 0, 0]:
            padding = "VALID"
        else:
            padding = "SAME"
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_dy = tf.compat.v1.placeholder(out_backprop.dtype, shape=out_backprop.shape)
        tf_dw_result = tf.compat.v1.nn.depthwise_conv2d_backprop_filter(tensor_x, w_shape, tensor_dy,
                                                                        strides=[1, strideh, stridew, 1],
                                                                        padding=padding, data_format="NHWC",
                                                                        dilations=dilations)
        feed_dict = {tensor_x: x, tensor_dy: out_backprop}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(tf_dw_result, feed_dict=feed_dict)
        output = _native_fun(out.shape, out)
    else:
        filter_size = (kh, kw, k_c, multi)
        output = _gen_depthwise_conv2d_backprop_filter_data(x, out_backprop, filter_size, strides=[1, strideh,
                                                                                                   stridew,
                                                                                                   1], pads=pads)
    return output


@register_golden(["conv2d_backprop_input_drelu"])
def _conv2d_backprop_input_drelu(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    from tensorflow.python.ops import gen_nn_ops
    _, conv_filter, out_backprop, input_mask = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    output_dtype = context.output_dtypes
    if groups != 1:
        if data_format == "NCHW":
            N, C, H, W = input_size
        else:
            N, H, W, C = input_size
        C1 = (C + 15) // 16
        return np.zeros((N, C1, H, W, 16))
    # 5HD input only
    if len(out_backprop.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    # dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = out_backprop.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    if data_format == "NCHW":
        N, C, H, W = input_size
    else:
        N, H, W, C = input_size
    # filter to NHWC
    out_backprop = out_backprop.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    C = (C + 15) // 16 * 16
    x_shape = (N, H, W, C)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    # NC1HW2 to 5HD to NHWC
    input_mask = np.array(input_mask)

    def _f_to_t(tmp):
        two_num_1 = bin(tmp[0]).split("0b")[1][::-1]
        two_num_1 += '0' * (8 - len(two_num_1))
        two_num_2 = bin(tmp[1]).split("0b")[1][::-1]
        two_num_2 += '0' * (8 - len(two_num_2))
        two_num = two_num_1 + two_num_2
        two_num_list = [int(x) for x in two_num]
        return two_num_list

    input_mask_data = np.zeros([N, C // 16, H, W, 16])
    for n in range(N):
        for c1 in range(C // 16):
            for h in range(H):
                for w in range(W):
                    input_mask_data[n, c1, h, w, :] = _f_to_t(input_mask[n, c1, h, w])
    ##################
    # def f_to_t(tmp):
    #     two_num = bin(tmp).split("0b")[1][::-1]
    #     two_num = two_num + '0' * (8 - len(two_num))
    #     two_num_list = [int(x) for x in two_num]
    #     return two_num_list

    input_mask_data = input_mask_data.transpose((0, 2, 3, 1, 4)).reshape((N, H, W, C)).astype(np.float16)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tensor_dy = tf.compat.v1.placeholder(out_backprop.dtype, shape=out_backprop.shape)
    tf_dx_result = tf.compat.v1.nn.conv2d_backprop_input(x_shape, tensor_filter, tensor_dy,
                                                         strides=[1, strideh, stridew, 1],
                                                         padding=((0, 0), (pad_top, pad_bottom),
                                                                  (pad_left, pad_right), (0, 0)),
                                                         data_format="NHWC", use_cudnn_on_gpu=False,
                                                         dilations=dilations)

    tf_dx_result = tf.compat.v1.cast(tf_dx_result, tf.float16)
    relu = tf.compat.v1.placeholder(tf_dx_result.dtype, shape=x_shape)
    mask = tf.compat.v1.nn.relu(relu)
    drelu = gen_nn_ops.relu_grad(tf_dx_result, mask, "drelu")

    feed_dict = {tensor_dy: out_backprop, tensor_filter: conv_filter, relu: input_mask_data}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(drelu, feed_dict=feed_dict)

    # NHWC to NC1HWC0
    output = out.reshape((N, H, W, C // 16, 16)).transpose(0, 3, 1, 2, 4)
    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output


@register_golden(["conv2d_backprop_input_vadd_drelu"])
def _conv2d_backprop_input_vadd_drelu(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    from tensorflow.python.ops import gen_nn_ops
    _, conv_filter, out_backprop, add_tensor, input_mask = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    if groups != 1:
        N, C, H, W = input_size
        C1 = (C + 15) // 16
        return np.zeros((N, C1, H, W, 16))
    # 5HD input only
    if len(out_backprop.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    # dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = out_backprop.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    if data_format == "NCHW":
        N, C, H, W = input_size
    else:
        N, H, W, C = input_size
    # filter to NHWC
    out_backprop = out_backprop.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    C = (C + 15) // 16 * 16
    x_shape = (N, H, W, C)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)

    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tensor_dy = tf.compat.v1.placeholder(out_backprop.dtype, shape=out_backprop.shape)
    tf_dx_result = tf.compat.v1.nn.conv2d_backprop_input(x_shape, tensor_filter, tensor_dy,
                                                         strides=[1, strideh, stridew, 1],
                                                         padding=((0, 0), (pad_top, pad_bottom),
                                                                  (pad_left, pad_right), (0, 0)),
                                                         data_format="NHWC", use_cudnn_on_gpu=False,
                                                         dilations=dilations)

    add_ = tf.compat.v1.placeholder(add_tensor.dtype, shape=x_shape)
    tf_dx_result = tf.compat.v1.cast(tf_dx_result, tf.float16)
    dx_vadd = tf.compat.v1.math.add(tf_dx_result, add_)
    relu = tf.compat.v1.placeholder(tf_dx_result.dtype, shape=x_shape)
    mask = tf.compat.v1.nn.relu(relu)
    dx_vadd_drelu = gen_nn_ops.relu_grad(dx_vadd, mask, "drelu")
    # NC1HWC0 to NHWC
    add_tensor = add_tensor.transpose(0, 2, 3, 1, 4).reshape(N, H, W, C)

    # NC1HW2 to 5HD to NHWC
    def _f_to_t(tmp):
        two_num_1 = bin(tmp[0]).split("0b")[1][::-1]
        two_num_1 += '0' * (8 - len(two_num_1))
        two_num_2 = bin(tmp[1]).split("0b")[1][::-1]
        two_num_2 += '0' * (8 - len(two_num_2))
        two_num = two_num_1 + two_num_2
        two_num_list = [int(x) for x in two_num]
        return two_num_list

    input_mask_data = np.zeros([N, C // 16, H, W, 16])
    for n in range(N):
        for c1 in range(C // 16):
            for h in range(H):
                for w in range(W):
                    input_mask_data[n, c1, h, w, :] = _f_to_t(input_mask[n, c1, h, w])
    input_mask_data = input_mask_data.transpose((0, 2, 3, 1, 4)).reshape((N, H, W, C)).astype(np.float16)
    feed_dict = {tensor_dy: out_backprop, tensor_filter: conv_filter, add_: add_tensor, relu: input_mask_data}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(dx_vadd_drelu, feed_dict=feed_dict)

    # NHWC to NC1HWC0
    output = out.reshape((N, H, W, C // 16, 16)).transpose(0, 3, 1, 2, 4)
    if output_dtype[0] == 'float16':
        output = due_overflow(output.astype(np.float16))
    return output
