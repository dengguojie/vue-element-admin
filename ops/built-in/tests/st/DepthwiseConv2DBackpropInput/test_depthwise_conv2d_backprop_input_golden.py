#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import numpy as np
import tensorflow as tf


def calc_expect_func_dx(filter, y, strides, out_backprop=None, input_size=None, x=None,
                     pads=None, dilations=None, groups=1, data_format='NCHW', output_padding=None,
                     offset_x=0, padding=None, bias=None):
    weight = filter
    filter_data = weight.get('value')
    filter_shape = filter_data.shape
    # filter_shape = weight.get('shape')
    filter_dtype = weight.get('dtype')
    filter_format = weight.get("format")
    if not out_backprop:
        out_backprop = x
    dy_data = out_backprop.get('value')
    dy_shape = dy_data.shape
    # dy_shape = dy.get('shape')
    dy_dtype = out_backprop.get('dtype')
    input_size_shape = y.get('shape')
    y_dtype = y.get('dtype')
    y_format = y.get('format')
    print('------------params:', filter_shape,
          dy_shape, y, strides, pads, groups, data_format)

    if filter_dtype == 'float16':
        filter_dtype = 'float32'
    if dy_dtype == 'float16':
        dy_dtype = 'float32'
    if data_format == 'NHWC':
        dy = dy_data.astype(dy_dtype)
    else:
        dy = dy_data.transpose(0, 2, 3, 1).astype(dy_dtype)

    if filter_format == "HWCN":
        w = filter_data.astype(filter_dtype)
    elif filter_format == "NCHW":
        w = filter_data.transpose(2, 3, 1, 0).astype(filter_dtype)
    else:
        w = filter_data.transpose(1, 2, 3, 0).astype(filter_dtype)

    if y_format == 'NCHW':
        Ni, Ci, Hi, Wi = input_size_shape
    else:
        Ni, Hi, Wi, Ci = input_size_shape

    h_index = data_format.index('H')
    w_index = data_format.index('W')
    if len(strides) == 2:
        strideh, stridew = strides
    else:
        strideh, stridew = strides[h_index], strides[w_index]
    if dilations is None:
        dilations = (1, 1, 1, 1)
    dilationh, dilationw = dilations[h_index], dilations[w_index]

    if pads is None:
        pads = _getPads(padding, [Ni, Hi, Wi, Ci], w.shape, dy.shape,
                        (strideh, stridew), (dilationh, dilationw))
    pad_top, pad_bottom, pad_left, pad_right = pads

    if (dilationh, dilationw) == (1, 1):
        if groups == 1:
            tensor_filter = tf.compat.v1.placeholder(w.dtype, shape=w.shape)
            tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
            tf_dx = tf.nn.conv2d_backprop_input(
                [Ni, Hi, Wi, Ci],
                tensor_filter,
                tensor_dy,
                strides=[1, strideh, stridew, 1],
                padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                data_format='NHWC',
                use_cudnn_on_gpu=False,
                dilations=[1, dilationh, dilationw, 1])

            feed_dict = {tensor_filter: w, tensor_dy: dy}
            if bias is not None:
                bias_data = bias.get("value").astype(dy_dtype)
                tensor_bias = tf.compat.v1.placeholder(bias_data.dtype,
                                                        shape=bias_data.shape)
                tf_dx = tf.nn.bias_add(tf_dx, tensor_bias)
                feed_dict[tensor_bias] = bias_data
            init_op = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:
                sess.run(init_op)
                out = sess.run(tf_dx, feed_dict=feed_dict)
        else:
            dy_split_shape, w_split_shape = list(dy.shape), list(w.shape)
            w_split_shape[3] = w_split_shape[3] // groups
            dy_split_shape[3] = dy_split_shape[3] // groups
            tensor_filter = tf.compat.v1.placeholder(w.dtype, shape=w_split_shape)
            tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy_split_shape)
            ci_split = Ci // groups
            tf_dx = tf.nn.conv2d_backprop_input(
                [Ni, Hi, Wi, ci_split],
                tensor_filter,
                tensor_dy,
                strides=[1, strideh, stridew, 1],
                padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                data_format='NHWC',
                use_cudnn_on_gpu=False,
                dilations=[1, dilationh, dilationw, 1])
            bias_data = None
            tensor_bias = None
            bias_split_shape = None
            if bias is not None:
                bias_data = bias.get("value").astype(dy_dtype)
                bias_split_shape = list(bias_data.shape)
                bias_split_shape[0] = bias_split_shape[0] // groups
                tensor_bias = tf.compat.v1.placeholder(bias_data.dtype,
                                                        shape=bias_split_shape)
                tf_dx = tf.nn.bias_add(tf_dx, tensor_bias)
            init_op = tf.compat.v1.global_variables_initializer()
            for i in range(groups):
                feed_dict = {
                    tensor_filter:
                    w[:, :, :, (w_split_shape[3] * i): (w_split_shape[3] * i +
                                                    w_split_shape[3])],
                    tensor_dy:
                    dy[:, :, :, (dy_split_shape[3] * i): (dy_split_shape[3] * i +
                                                    dy_split_shape[3])]
                }
                if bias is not None:
                    feed_dict[tensor_bias] = bias_data[bias_split_shape[0] *
                                                    i: bias_split_shape[0] *
                                                    (i + 1)]
                with tf.compat.v1.Session() as sess:
                    sess.run(init_op)
                    if i == 0:
                        out = sess.run(tf_dx, feed_dict=feed_dict)
                    else:
                        out = np.concatenate((out, sess.run(tf_dx, feed_dict=feed_dict)), axis=3)
        if y_format == "NCHW":
            out = out.transpose(0, 3, 1, 2).copy()
        print('------golden:', out.shape)
        res = out.astype(y_dtype)

    else:
        import torch
        if data_format == "NHWC":
            dy_data = np.transpose(dy_data, (0, 3, 1, 2))
        dilation_torch = [dilationh, dilationw]
        strides_torch = [strideh, stridew]
        input_size_torch = [Ni, Ci, Hi, Wi]

        if filter_format == "HWCN":
            filter_data = np.transpose(filter_data, (3, 2, 0, 1))
        elif filter_format == "NHWC":
            filter_data = np.transpose(filter_data, (0, 3, 1, 2))

        padding_torch = [min(pads[0], pads[1]), min(pads[2], pads[3])]
        dy_data_nchw = dy_data.astype(np.float32)
        filter_data_nchw = filter_data.astype(np.float32)

        out_backprop_hw = dy_data_nchw.shape[2:]
        input_size_hw = input_size_torch[2:]
        w_h, w_w = filter_data_nchw.shape[2:]
        filter_h_dilation = (w_h - 1) * dilation_torch[0] + 1
        filter_w_dilation = (w_w - 1) * dilation_torch[1] + 1
        filter_hw = [filter_h_dilation, filter_w_dilation]
        output_padding = list(o - (i - 1) * s + 2 * p - k for o, i, s, p, k in
                            zip(input_size_hw,
                                out_backprop_hw,
                                strides_torch,
                                padding_torch,
                                filter_hw))
        f = lambda x: [max(x[0], 0), max(x[1], 0)]
        output_padding = f(output_padding)
        print("output_padding:", output_padding)

        out = torch.nn.functional.conv_transpose2d(torch.from_numpy(dy_data_nchw), torch.from_numpy(filter_data_nchw), bias=None, stride=strides_torch, padding=padding_torch,
                                                output_padding=output_padding, dilation=dilation_torch, groups=groups).numpy()

        res = np.zeros(input_size_torch, dtype=np.float32)
        res[:,:,:,:] = out[:, :, :input_size_torch[2], :input_size_torch[3]]

        if y_dtype == "float16":
            res = np.maximum(res, -65504)
            res = np.minimum(res, 65504)
            res = res.astype(np.float16)
        if data_format == "NHWC":
            res = np.transpose(res, (0, 2, 3 , 1))

        if bias is not None:
            bias_data = bias.get("value")
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True))
            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            bias_res = tf.compat.v1.nn.bias_add(res, bias_data, data_format)
            res = sess.run(bias_res)
            sess.close()
            if y_dtype == "float16":
                res = np.maximum(res, -65504)
                res = np.minimum(res, 65504)
                res = res.astype(np.float16)

    return [res]

def calc_expect_func(input_size,
                    weight,
                    out_backprop,
                    input_grad,
                    strides,
                    pads=None,
                    dilations=None,
                    data_format='NCHW',
                    padding=None):
    filter = weight
    filter_data = weight.get("value")
    filter_format = weight.get("format")
    filter_shape = filter_data.shape
    if filter_format == 'HWCN':
        kh, kw, kc, kn = filter_shape
        filter_data = filter_data.reshape(kh, kw, 1, kc*kn)
        filter["value"] = filter_data
    elif filter_format == 'NCHW':
        kn, kc, kh, kw = filter_shape
        filter_data = filter_data.transpose(1, 0, 2, 3).reshape(kc*kn, 1, kh, kw)
        filter["value"] = filter_data
    c_index = input_grad.get("format").index("C")
    groups = input_grad.get("shape")[c_index]
    return calc_expect_func_dx(filter,
                            input_grad,
                            strides,
                            out_backprop=out_backprop,
                            input_size=input_size,
                            pads=pads,
                            dilations=dilations,
                            groups=groups,
                            data_format=data_format,
                            padding=padding)

def _getPads(padding, x_shape, w_shape, dy_shape, strides, dilations):
    _, H, W, _ = x_shape
    kh, kw, _, _ = w_shape
    strideh, stridew = strides
    dilationh, dilationw = dilations
    He = (kh - 1) * dilationh + 1
    We = (kw - 1) * dilationw + 1
    if padding == 'VALID':
        pads = [0, 0, 0, 0]
    elif padding == 'SAME':
        if dy_shape is None:
            Ho = (H + strideh - 1) // strideh
            Wo = (W + stridew - 1) // stridew
        else:
            _, Ho, Wo, _ = dy_shape
        padh = max(0, (Ho - 1) * strideh + He - H)
        padw = max(0, (Wo - 1) * stridew + We - W)
        pads = [padh // 2, padh - padh // 2, padw // 2, padw - padw // 2]
    else:
        raise RuntimeError('not support this padding yet')

    return pads