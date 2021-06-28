#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import numpy as np
import tensorflow as tf


def calc_expect_func(x, filter_size, out_backprop, y, strides, pads=None,
                     dilations=None, groups=1, data_format='NCHW',
                     padding=None):
    fmap_data = x.get('value')
    fmap_shape = fmap_data.shape
    # fmap_shape = x.get('shape')
    fmap_dtype = x.get('dtype')
    dy_data = out_backprop.get('value')
    dy_shape = dy_data.shape
    # dy_shape = out_backprop.get('shape')
    dy_dtype = out_backprop.get('dtype')
    filter_shape = y.get('shape')
    y_dtype = y.get('dtype')
    y_format = y.get('format')
    print('------------params:', fmap_shape, dy_shape,
          y, strides, pads, groups, data_format)

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
        filter_n, filter_c, kh, kw = filter_shape
    elif y_format == 'NHWC':
        filter_n, kh, kw, filter_c = filter_shape
    else:
        kh, kw, filter_c, filter_n = filter_shape
        
    if pads is None:
        pads = _getPads(padding, x.shape, [kh, kw, filter_c, filter_n],
                        dy.shape, (strideh, stridew), (dilationh, dilationw))
    pad_top, pad_bottom, pad_left, pad_right = pads

    cin_ori = x.shape[3]
    if groups == 1:
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        tf_dw_result = tf.nn.conv2d_backprop_filter(tensor_x,
                                                    [kh, kw, filter_c, filter_n],
                                                    tensor_dy,
                                                    strides=[
                                                        1, strideh, stridew, 1],
                                                    padding=(
                                                        (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                                    data_format='NHWC',
                                                    use_cudnn_on_gpu=False,
                                                    dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_x: x, tensor_dy: dy}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(tf_dw_result, feed_dict=feed_dict)
    elif groups == cin_ori and strideh == stridew:
        if padding is None:
            padding = _getPadding(pads, x.shape, [kh, kw, filter_c, filter_n],
                                  dy.shape, (strideh, stridew),
                                  [dilationh, dilationw])
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        tf_dw_result = tf.nn.depthwise_conv2d_backprop_filter(tensor_x,
                                                              [kh, kw, filter_n,
                                                                  filter_c],
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
        out = out.transpose(0, 1, 3, 2)
    else:
        out = _conv2d_backprop_filter(x, [kh, kw, filter_c, filter_n], dy, [
                                      1, strideh, stridew, 1], pads, groups)

    if y_format == 'NCHW':
        out = out.transpose(3, 2, 0, 1).copy()
    elif y_format == 'NHWC':
        out = out.transpose(3, 0, 1, 2).copy()
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


def _conv2d_backprop_filter(x, filter_size, dy, strides, pads, groups):
    n, hi, wi, ci = x.shape
    kh, kw, filter_c, filter_n = filter_size
    n, ho, wo, co = dy.shape
    _, strideh, stridew, _ = strides
    pad_top, pad_bottom, pad_left, pad_right = pads

    group_dict = _calculate_group(ci, co, groups)
    real_g = group_dict.get('real_g')
    cin_g = group_dict.get('cin_g')
    # cin1_g = group_dict.get('cin1_g')
    cout_g = group_dict.get('cout_g')

    # dedy
    dedy_target = np.zeros((real_g, n, cout_g, ho, wo), dtype=dy.dtype)
    for i in range(co):
        dedy_target[i // cout_g, :, i % cout_g, :, :] = dy[:, :, :, i]

    # dedx
    dedx_target = np.zeros((real_g, n, cin_g, hi + pad_top +
                           pad_bottom, wi + pad_left + pad_right), dtype=x.dtype)
    for i in range(ci):
        dedx_target[i // cin_g, :, i % cin_g, pad_top:pad_top +
                    hi, pad_left:pad_left + wi] = x[:, :, :, i]

    tmp = np.zeros([real_g, n, cin_g, ho, kh, wo, kw])
    for j0 in range(ho):
        for j1 in range(kh):
            for k0 in range(wo):
                for k1 in range(kw):
                    tmp[:, :, :, j0, j1, k0, k1] = dedx_target[:,
                                                               :, :, j0 * strideh + j1, k0 * stridew + k1]
    dedx_target = tmp.transpose(0, 1, 2, 4, 6, 3, 5).reshape(real_g,
                                                             n, cin_g * kh * kw, ho, wo).copy()

    out = np.zeros([real_g, cin_g * kh * kw, cout_g])
    for i in range(real_g):
        for j in range(cin_g * kh * kw):
            for k in range(cout_g):
                out[i, j, k] = np.sum(dedy_target[i, :, k, :, :] *
                                      dedx_target[i, :, j, :, :])
    out = out.reshape(real_g * cin_g, kh, kw, cout_g).transpose(1, 2, 0, 3)
    return out
