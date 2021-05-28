#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(orig_input_shape, input_grad, out_grad, ksize, strides, padding, data_format='NCHW'):
    dy_data = input_grad.get('value')
    dy_shape = dy_data.shape
    # dy_shape = input_grad.get('shape')
    dy_dtype = input_grad.get('dtype')
    fmap_shape = out_grad.get('shape')
    y_dtype = out_grad.get('dtype')
    y_format = out_grad.get('format')
    print('------------params:', dy_shape, out_grad,
          ksize, strides, padding, data_format)

    h_index = data_format.index('H')
    w_index = data_format.index('W')
    strideh, stridew = strides[h_index], strides[w_index]
    ksize_h, ksize_w = ksize[h_index], ksize[w_index]

    if dy_dtype == 'float16':
        dy_dtype = 'float32'
    if data_format == 'NCHW':
        dy = dy_data.transpose(0, 2, 3, 1).astype(dy_dtype)
    else:
        dy = dy_data.astype(dy_dtype)
    if y_format == 'NCHW':
        N, C, H, W = fmap_shape
    else:
        N, H, W, C = fmap_shape

    tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
    avgpoolgrad_result = tf.raw_ops.AvgPoolGrad(orig_input_shape=[N, H, W, C], grad=tensor_dy, ksize=[1, ksize_h, ksize_w, 1],
                                                strides=[1, strideh, stridew, 1], padding=padding, data_format='NHWC')
    feed_dict = {tensor_dy: dy}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        out = sess.run(avgpoolgrad_result, feed_dict=feed_dict)

    if y_format == 'NCHW':
        out = out.transpose(0, 3, 1, 2).copy()
    print('------golden:', out.shape)
    res = out.astype(y_dtype)
    return [res]
