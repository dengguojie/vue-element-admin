#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import numpy as np
import tensorflow as tf


def calc_expect_func(input_size, weight, out_backprop, input_grad, strides,
                     pads=None, dilations=None, data_format='NCHW',
                     padding=None):
    filter_data = weight.get('value')
    filter_shape = filter_data.shape
    # filter_shape = weight.get('shape')
    filter_dtype = weight.get('dtype')
    dy_data = out_backprop.get('value')
    dy_shape = dy_data.shape
    # dy_shape = dy.get('shape')
    dy_dtype = out_backprop.get('dtype')
    input_size_shape = input_grad.get('shape')
    y_dtype = input_grad.get('dtype')
    y_format = input_grad.get('format')
    print('------------params:', filter_shape,
          dy_shape, input_grad, strides, pads, data_format)

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
        w = filter_data.transpose(2, 3, 0, 1).astype(
            filter_dtype)  # NCHW->HWCN
        dy = dy_data.transpose(0, 2, 3, 1).astype(dy_dtype)

    if y_format == 'NCHW':
        Ni, Ci, Hi, Wi = input_size_shape
    else:
        Ni, Hi, Wi, Ci = input_size_shape

    if strideh == stridew:
        if padding is None:
            padding = _getPadding(pads, [Ni, Hi, Wi, Ci], w.shape, dy.shape,
                                  (strideh, stridew), [dilationh, dilationw])
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
        if pads is None:
            pads = _getPads(padding, [Ni, Hi, Wi, Ci], w.shape, dy.shape,
                        (strideh, stridew), (dilationh, dilationw))
        # pad_top, pad_bottom, pad_left, pad_right = pads
        out = _depthwise_conv2d_native_backprop_input([Ni, Hi, Wi, Ci], w, dy,
                                                      [1, strideh, stridew, 1], pads)

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


def _conv2d(input_, filter_, strides=None):
    if strides is None:
        strides = [1, 1]
    ish = input_.shape
    fsh = filter_.shape
    strideh, stridew = strides
    Ho = (ish[1] - fsh[0]) // strideh + 1
    Wo = (ish[2] - fsh[1]) // stridew + 1
    osh = [ish[0], Ho, Wo, fsh[3]]
    output = np.zeros(osh)
    for p in range(osh[0]):
        for i in range(osh[1]):
            for j in range(osh[2]):
                for di in range(fsh[0]):
                    for dj in range(fsh[1]):
                        t = np.dot(
                            input_[p, strideh * i + di, stridew * j + dj, :],
                            filter_[di, dj, :, :])
                        output[p, i, j] = np.sum([t, output[p, i, j]], axis=0)
    return output


def _depthwise_conv2d_native_backprop_input(input_size, w, dy, strides, pads):
    Ni, Hi, Wi, Ci = input_size
    kh, kw, filtesr_c, filter_n = w.shape
    No, Ho, Wo, Co = dy.shape
    _, strideh, stridew, _ = strides
    pad_top, pad_bottom, pad_left, pad_right = pads

    full_height = Hi + kh - 1
    full_width = Wi + kw - 1
    dilated_height = Ho * strideh - (strideh - 1)
    dilated_width = Wo * stridew - (stridew - 1)
    pad_out_top = kh - 1 - pad_top
    pad_out_bottom = full_height - dilated_height - pad_out_top
    pad_out_left = kw - 1 - pad_left
    pad_out_right = full_width - dilated_width - pad_out_left
    padded_dilated_height = dilated_height + pad_out_top + pad_out_bottom
    padded_dilated_width = dilated_width + pad_out_left + pad_out_right

    padded_dilated_grad = np.zeros(
        [Ni, padded_dilated_height, padded_dilated_width, Co])
    for i in range(Ho):
        index_h = pad_out_top + i * strideh
        for j in range(Wo):
            index_w = pad_out_left + j * stridew
            padded_dilated_grad[:, index_h, index_w, :] = dy[:, i, j, :]

    filter_rotated = np.zeros([kh, kw, Ci, 1])
    for i in range(kh):
        for j in range(kw):
            filter_rotated[kh - 1 - i, kw - 1 - j, :, :] = w[i, j, :, :]

    input_grad = np.zeros([Ni, Hi, Wi, Ci])
    # used as fmap
    padded_dilated_grad_piece = np.zeros(
        [Ni, padded_dilated_height, padded_dilated_width, 1])
    # used as filter
    filter_rotated_piece = np.zeros([kh, kw, 1, 1])
    for i in range(Ci):
        padded_dilated_grad_piece[:, :, :, 0] = padded_dilated_grad[:, :, :, i]
        filter_rotated_piece[:, :, 0, 0] = filter_rotated[:, :, i, 0]
        input_grad[:, :, :, i] = _conv2d(
            padded_dilated_grad_piece, filter_rotated_piece).reshape([Ni, Hi, Wi])

    return input_grad
