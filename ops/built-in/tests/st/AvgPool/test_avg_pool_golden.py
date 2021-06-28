#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(x, y, ksize, strides, padding, data_format='NCHW', offset_x=0):
    fmap_data = x.get('value')
    fmap_shape = fmap_data.shape
    # fmap_shape = x.get('shape')
    fmap_dtype = x.get('dtype')
    y_dtype = y.get('dtype')
    y_format = y.get('format')
    print('------------params:', fmap_shape, y,
          ksize, strides, padding, data_format)

    h_index = data_format.index('H')
    w_index = data_format.index('W')
    strideh, stridew = strides[h_index], strides[w_index]
    ksize_h, ksize_w = ksize[h_index], ksize[w_index]

    if fmap_dtype == 'float16':
        fmap_dtype = 'float32'
    if data_format == 'NCHW':
        x = fmap_data.transpose(0, 2, 3, 1).astype(fmap_dtype)
    else:
        x = fmap_data.astype(fmap_dtype)

    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    avg_pool_result = tf.nn.avg_pool(tensor_x, ksize=[1, ksize_h, ksize_w, 1],
                                     strides=[1, strideh, stridew, 1],
                                     padding=padding, data_format='NHWC')
    feed_dict = {tensor_x: x}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        out = sess.run(avg_pool_result, feed_dict=feed_dict)

    if y_format == 'NCHW':
        out = out.transpose(0, 3, 1, 2).copy()
    print('------golden:', out.shape)
    res = out.astype(y_dtype)
    return [res]
