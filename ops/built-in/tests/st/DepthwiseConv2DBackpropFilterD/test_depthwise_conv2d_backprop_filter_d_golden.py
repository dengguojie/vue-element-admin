#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import numpy as np
import tensorflow as tf


def calc_expect_func(x, out_backprop, filter_grad, filter_size, strides,
                     pads=None, dilations=None, data_format='NCHW',
                     padding=None):
    fmap_data = x.get('value')
    fmap_shape = fmap_data.shape
    # fmap_shape = x.get('shape')
    fmap_dtype = x.get('dtype')
    dy_data = out_backprop.get('value')
    dy_shape = dy_data.shape
    # dy_shape = out_backprop.get('shape')
    dy_dtype = out_backprop.get('dtype')
    filter_shape = filter_grad.get('shape')
    # y_dtype = filter_grad.get('dtype')
    y_format = filter_grad.get('format')
    print('------------params:', fmap_shape, dy_shape,
          filter_grad, filter_size, strides, pads, data_format)

    h_index = data_format.index('H')
    w_index = data_format.index('W')
    strideh, stridew = strides[h_index], strides[w_index]
    if dilations is None:
        dilations = (1, 1, 1, 1)
    dilationh, dilationw = dilations[h_index], dilations[w_index]

    if fmap_dtype == 'float16':
        fmap_dtype = 'float32'
    if dy_dtype == 'float16':
        dy_dtype = 'float32'
    if data_format == 'NCHW':
        x = fmap_data.transpose(0, 2, 3, 1).astype(fmap_dtype)
        dy = dy_data.transpose(0, 2, 3, 1).astype(dy_dtype)
    else:
        x = fmap_data.astype(fmap_dtype)
        dy = dy_data.astype(dy_dtype)
    if y_format == 'NCHW':
        k, Ci, kh, kw = filter_shape
    elif y_format == 'NHWC':
        k, kh, kw, Ci = filter_shape
    else:
        kh, kw, Ci, k = filter_shape

    if strideh == stridew:
        if padding is None:
            padding = _getPadding(pads, x.shape, [kh, kw, Ci, k], dy.shape,
                                  (strideh, stridew), [dilationh, dilationw])
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        tf_dw_result = tf.nn.depthwise_conv2d_backprop_filter(tensor_x,
                                                              [kh, kw, Ci, k],
                                                              tensor_dy,
                                                              strides=[
                                                                  1, strideh, stridew, 1],
                                                              padding=padding,
                                                              data_format='NHWC',
                                                              dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_x: x, tensor_dy: dy}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(tf_dw_result, feed_dict=feed_dict)
    else:
        if pads is None:
            pads = _getPads(padding, x.shape, [kh, kw, Ci, k], dy.shape,
                        (strideh, stridew), (dilationh, dilationw))
        # pad_top, pad_bottom, pad_left, pad_right = pads
        out = _depthwise_conv2d_native_backprop_filter(
            x, [kh, kw, Ci, k], dy, [1, strideh, stridew, 1], pads)

    if y_format == 'NCHW':
        out = out.transpose(3, 2, 0, 1).copy()
    elif y_format == 'NHWC':
        out = out.transpose(3, 0, 1, 2).copy()
    print('------golden:', out.shape)
    res = out.astype('float32')
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


def _depthwise_conv2d_native_backprop_filter(x, filter_size, dy, strides, pads):
    N, H, W, C = x.shape
    No, Ho, Wo, Co = dy.shape
    kh, kw, filter_c, filter_n = filter_size
    _, strideh, stridew, _ = strides
    pad_top, pad_bottom, pad_left, pad_right = pads

    dilated_height = Ho * strideh - (strideh - 1)
    dilated_width = Wo * stridew - (stridew - 1)
    if tuple(pads) == (0, 0, 0, 0):
        ori_padh = (Ho - 1) * strideh + kh - H
        ori_padw = (Wo - 1) * stridew + kw - W
        if ori_padh < 0:
            dilated_height -= ori_padh
        if ori_padw < 0:
            dilated_width -= ori_padw
    dilated_grad = np.zeros([N, dilated_height, dilated_width, Co])
    for i in range(Ho):
        index_h = i * strideh
        for j in range(Wo):
            index_w = j * stridew
            dilated_grad[:, index_h, index_w, :] = dy[:, i, j, :]

    filter_grad_tmp = np.zeros([N, kh, kw, C])
    # used as fmap
    input_piece = np.zeros(
        [1, H + pad_top + pad_bottom, W + pad_left + pad_right, 1])
    # used as filter
    dilated_grad_piece = np.zeros(
        [dilated_height, dilated_width, 1, 1])

    h_start = pad_top
    h_end = pad_top + H
    w_start = pad_left
    w_end = pad_left + W
    for j in range(N):
        for i in range(C):
            input_piece[0, h_start:h_end, w_start:w_end, 0] = x[j, :, :, i]
            dilated_grad_piece[:, :, 0, 0] = dilated_grad[j, :, :, i]
            filter_grad_tmp[j, :, :, i] = _conv2d(
                input_piece, dilated_grad_piece).reshape([kh, kw])
    out = np.sum(filter_grad_tmp, axis=0).reshape(
        [filter_n, kh, kw, filter_c]).transpose(1, 2, 3, 0)

    return out
