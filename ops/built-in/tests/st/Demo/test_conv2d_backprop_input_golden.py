#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import numpy as np
import tensorflow as tf


def calc_expect_func(input_size, weight, out_backprop, y, strides, pads=None,
                     dilations=None, groups=1, data_format='NCHW',
                     padding=None):
    filter_data = weight.get('value')
    filter_shape = filter_data.shape
    # filter_shape = weight.get('shape')
    filter_dtype = weight.get('dtype')
    dy_data = out_backprop.get('value')
    dy_shape = dy_data.shape
    # dy_shape = out_backprop.get('shape')
    dy_dtype = out_backprop.get('dtype')
    input_size_shape = y.get('shape')
    y_dtype = y.get('dtype')
    y_format = y.get('format')
    print('------------params:', filter_shape, dy_shape,
          y, strides, pads, groups, data_format)

    h_index = data_format.index('H')
    w_index = data_format.index('W')
    strideh, stridew = strides[h_index], strides[w_index]
    if dilations is None:
        dilations = (1, 1, 1, 1)
    dilationh, dilationw = dilations[h_index], dilations[w_index]

    if filter_dtype == 'float16':
        filter_dtype = 'float32'
    if dy_dtype == 'float16':
        dy_dtype = 'float32'
    if data_format == 'NHWC':
        w = filter_data.astype(filter_dtype)
        dy = dy_data.astype(dy_dtype)
    else:
        w = filter_data.transpose(2, 3, 1, 0).astype(filter_dtype)
        dy = dy_data.transpose(0, 2, 3, 1).astype(dy_dtype)

    if y_format == 'NCHW':
        Ni, Ci, Hi, Wi = input_size_shape
    else:
        Ni, Hi, Wi, Ci = input_size_shape
    if pads is None:
        pads = _getPads(padding, [Ni, Hi, Wi, Ci], w.shape, dy.shape,
                    (strideh, stridew), (dilationh, dilationw))
    pad_top, pad_bottom, pad_left, pad_right = pads

    if groups == 1:
        tensor_filter = tf.compat.v1.placeholder(w.dtype, shape=w.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        tf_dx = tf.nn.conv2d_backprop_input([Ni, Hi, Wi, Ci],
                                            tensor_filter,
                                            tensor_dy,
                                            strides=[1, strideh, stridew, 1],
                                            padding=(
                                                (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                            data_format='NHWC',
                                            use_cudnn_on_gpu=False,
                                            dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_filter: w, tensor_dy: dy}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(tf_dx, feed_dict=feed_dict)
    elif groups == Ci and strideh == stridew:
        if padding is None:
            padding = _getPadding(pads, [Ni, Hi, Wi, Ci], w.shape, dy.shape,
                                  (strideh, stridew), [dilationh, dilationw])
        w = w.transpose(0, 1, 3, 2)
        tensor_filter = tf.compat.v1.placeholder(w.dtype, shape=w.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        dx = tf.nn.depthwise_conv2d_backprop_input([Ni, Hi, Wi, Ci],
                                                   tensor_filter,
                                                   tensor_dy,
                                                   strides=[
                                                       1, strideh, stridew, 1],
                                                   padding=padding,
                                                   data_format='NHWC',
                                                   dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_filter: w, tensor_dy: dy}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(dx, feed_dict=feed_dict)
    else:
        out = _conv2d_backprop_input([Ni, Hi, Wi, Ci], w, dy,
                                     [1, strideh, stridew, 1], pads,
                                     [1, dilationh, dilationw, 1], groups)

    if y_format == 'NCHW':
        out = out.transpose(0, 3, 1, 2).copy()
    print('------golden:', out.shape)
    res = out.astype(y_dtype)
    return [res]


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


def _getPadding(pads, x_shape, w_shape, dy_shape, strides, dilations):
    padt, padb, padl, padr = pads
    _, H, W, _ = x_shape
    kh, kw, _, _ = w_shape
    strideh, stridew = strides
    dilationh, dilationw = dilations
    He = (kh - 1) * dilationh + 1
    We = (kw - 1) * dilationw + 1

    if dy_shape is None:
        if padt != 0 or padb != 0 or padl != 0 or padr != 0:
            Ho = (H + strideh - 1) // strideh
            Wo = (W + stridew - 1) // stridew
            if padt + padb == max(0, (Ho - 1) * strideh + He - H) and \
                    padl + padr == max(0, (Wo - 1) * stridew + We - W):
                padding = 'SAME'
            else:
                padding = 'CALCULATED'
                raise RuntimeError('not support this padding yet')
        else:
            Ho = (H - He) // strideh + 1
            Wo = (W - We) // stridew + 1
            padding = 'VALID'
    else:
        _, Ho, Wo, _ = dy_shape
        if Ho == (H + strideh - 1) // strideh and \
                Wo == (W + stridew - 1) // stridew and \
                padt + padb == max(0, (Ho - 1) * strideh + He - H) and \
                padl + padr == max(0, (Wo - 1) * stridew + We - W):
            padding = 'SAME'
        elif Ho == (H - He) // strideh + 1 and \
                Wo == (W - We) // stridew + 1 and \
                padt == 0 and padb == 0 and padl == 0 and padr == 0:
            padding = 'VALID'
        else:
            padding = 'CALCULATED'
            raise RuntimeError('not support this padding yet')

    return padding


def _ceil(a, b):
    return (a + b - 1) // b


def _align(a, b):
    return _ceil(a, b) * b


def _lcm(param1, param2):
    tmp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, (param1 % param2)

    return tmp // param2


def _calculate_group(cin, cout, groups):
    block_size = 16

    mag_factor0 = _lcm(cin // groups, block_size) // (cin // groups)
    mag_factor1 = _lcm(cout // groups, block_size) // (cout // groups)
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin_g = _align(mag_factor * (cin // groups), block_size)
    cout_g = _align(mag_factor * (cout // groups), block_size)

    group_dict = {'real_g': _ceil(groups, mag_factor),
                  'mag_factor': mag_factor,
                  'cin_g': cin_g,
                  'cin1_g': cin_g // block_size,
                  'cout_g': cout_g,
                  'cout1_g': cout_g // block_size,
                  'groups': groups,
                  'cin_ori': cin // groups,
                  'cout_ori': cout // groups}
    return group_dict


def conv_backward_naive(dout, w, x, strides, pads):
    G, N, F, Ho, Wo = dout.shape
    _, N, C, H, W = x
    kh = w.shape[3]
    kw = w.shape[4]
    stride_h, stride_w = strides
    pad_up, pad_down, pad_left, pad_right = pads

    dx = np.zeros(x)
    dx_pad = np.pad(dx, [(0, 0), (0, 0), (0, 0), (pad_up, pad_down),
                         (pad_left, pad_right)], 'constant')
    for m in range(G):
        for n in range(N):
            for i in range(Ho):
                for j in range(Wo):
                    for f in range(F):
                        dx_pad[m, n, :, i * stride_h: i * stride_h + kh, j *
                               stride_w: j * stride_w + kw] += w[m, f] * dout[m, n, f, i, j]
    dx = dx_pad[:, :, :, pad_up: pad_up + H, pad_left: pad_left + W]
    return dx


def _conv2d_backprop_input(input_size, w, dy, strides, pads, dilations, groups):
    kh, kw, filter_c, filter_n = w.shape
    No, Ho, Wo, Co = dy.shape
    Ni, Hi, Wi, Ci = input_size
    _, strideh, stridew, _ = strides
    _, dilationh, dilationw, _ = dilations
    pad_top, pad_bottom, pad_left, pad_right = pads

    group_dict = _calculate_group(Ci, Co, groups)
    G = group_dict.get('real_g')
    ci_ori = group_dict.get('cin_ori')
    co_ori = group_dict.get('cout_ori')
    cin1_g = group_dict.get('cin1_g')
    cout1_g = group_dict.get('cout1_g')
    E = group_dict.get('mag_factor')

    w = w.transpose(3, 2, 0, 1)
    dy = dy.transpose(0, 3, 1, 2)
    filter_after = np.zeros(
        [G * cout1_g * 16, cin1_g * 16, kh, kw]).astype(w.dtype)
    for m in range(groups):
        for k in range(co_ori):
            for n in range(ci_ori):
                i = m // E
                j = m % E
                filter_after[i * E * co_ori + j * co_ori + k, j * ci_ori + n, :, :] = \
                    w[i * E * co_ori + j * co_ori + k, n, :, :]
    he = (kh - 1) * dilationh + 1
    we = (kw - 1) * dilationw + 1
    w_dilations_shape = [G * cout1_g * 16, cin1_g * 16, he, we]
    filter_dilations = np.zeros(w_dilations_shape).astype(w.dtype)
    for m in range(kh):
        for n in range(kw):
            filter_dilations[:, :, m * dilationh, n *
                             dilationw] = filter_after[:, :, m, n]

    dy_after_shape = [G, No, cout1_g * 16, Ho, Wo]
    dy_after = np.zeros(dy_after_shape).astype(dy.dtype)
    for i in range(G):
        for j in range(No):
            for k in range(co_ori):
                for n in range(Ho):
                    for m in range(Wo):
                        dy_after[i, j, k, n, m] = dy[j,
                                                     i * cout1_g * 16 + k, n, m]
    filter_dilations = filter_dilations.reshape(
        G, cout1_g * 16, cin1_g * 16, he, we)
    input_shape = G, Ni, cin1_g * 16, Hi, Wi
    dx_bp_before = conv_backward_naive(
        dy_after, filter_dilations, input_shape, [strideh, stridew], pads)
    out = np.zeros([Ni, Ci, Hi, Wi]).astype(dy.dtype)
    for j in range(ci_ori):
        out[:, j, :, :] = dx_bp_before[j // (cin1_g * 16),
                                       :, j % (cin1_g * 16), :, :]

    out = out.transpose(0, 2, 3, 1)
    return out
