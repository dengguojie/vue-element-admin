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

    h_index = data_format.index('H')
    w_index = data_format.index('W')
    strideh, stridew = strides[h_index], strides[w_index]
    if dilations is None:
        dilations = (1, 1, 1, 1)
    dilationh, dilationw = dilations[h_index], dilations[w_index]

    if pads is None:
        pads = _getPads(padding, x.shape, [kh, kw, filter_c, filter_n],
                        dy.shape, (strideh, stridew), (dilationh, dilationw))
    pad_top, pad_bottom, pad_left, pad_right = pads

    if (dilationh, dilationw) == (1, 1):
        dy_split_shape, x_split_shape = list(dy.shape), list(x.shape)
        x_split_shape[3] = x_split_shape[3] // groups
        dy_split_shape[3] = dy_split_shape[3] // groups
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x_split_shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy_split_shape)
        tf_dw_result = tf.nn.conv2d_backprop_filter(tensor_x,
                                                    [kh, kw, filter_c, filter_n // groups],
                                                    tensor_dy,
                                                    strides=[
                                                        1, strideh, stridew, 1],
                                                    padding=(
                                                        (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                                    data_format='NHWC',
                                                    use_cudnn_on_gpu=False,
                                                    dilations=[1, dilationh, dilationw, 1])
        init_op = tf.compat.v1.global_variables_initializer()
        for i in range(groups):
            feed_dict = {
                tensor_x:
                x[:, :, :, (x_split_shape[3] * i): (x_split_shape[3] * i + x_split_shape[3])],
                tensor_dy:
                dy[:, :, :, (dy_split_shape[3] * i): (dy_split_shape[3] * i + dy_split_shape[3])]
            }
            with tf.compat.v1.Session() as sess:
                sess.run(init_op)
                if i == 0:
                    out = sess.run(tf_dw_result, feed_dict=feed_dict)
                else:
                    out = np.concatenate(
                        (out, sess.run(tf_dw_result, feed_dict=feed_dict)),
                        axis = 3)

        if y_format == 'NCHW':
            out = out.transpose(3, 2, 0, 1).copy()
        elif y_format == 'NHWC':
            out = out.transpose(3, 0, 1, 2).copy()
    else:
        import torch
        from torch.autograd import Variable
        h_offset = pad_bottom - pad_top
        w_offset = pad_right - pad_left
        if data_format == 'NHWC':
            x = fmap_data.transpose(0, 3, 1, 2).astype(fmap_dtype)
            dy = dy_data.transpose(0, 3, 1, 2).astype(dy_dtype)
        else:
            x = fmap_data.astype(fmap_dtype)
            dy = dy_data.astype(dy_dtype)
        fmap_n, fmap_c, fmap_h, fmap_w = x.shape
        fmap_data = np.pad(x, ((0, 0), (0, 0), (0, h_offset), (0, w_offset)), 'constant', constant_values=(0, 0))
        fmap_data = np.reshape(fmap_data, [fmap_n, fmap_c, fmap_h + h_offset, fmap_w + w_offset])
        fmap_data = Variable(torch.from_numpy(fmap_data).type(torch.float32), requires_grad=True)
        weight_data = np.random.uniform(-1, 1, size=(filter_n, filter_c, kh, kw)).astype(np.float32)
        weight = Variable(torch.from_numpy(weight_data).type(torch.float32), requires_grad=True)
        grad_data = torch.from_numpy(dy).type(torch.float32)

        out = torch.nn.functional.conv2d(fmap_data, weight, stride=(strideh, stridew), padding=(pad_top, pad_left),
                                         dilation=(dilationh, dilationw), groups=groups)
        out.backward(grad_data, retain_graph=True)
        out = weight.grad.detach().numpy().astype(np.float32).copy()
        if y_format == 'HWCN':
            out = out.transpose(2, 3, 1, 0).copy()
        elif y_format == 'NHWC':
            out = out.transpose(0, 2, 3, 1).copy()

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