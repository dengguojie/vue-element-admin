#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(x, min_range, max_range, mode, y):
    x = x.get('value')
    min_range = min_range.get('value')[0]
    max_range = max_range.get('value')[0]

    tensor_x = tf.placeholder(x.dtype, shape= x.shape)
    tensor_x_qint = tf.cast(tensor_x, tf.qint32)

    tensor_min_range = tf.constant(min_range)
    tensor_max_range = tf.constant(max_range)
    out = tf.quantization.dequantize(tensor_x_qint, tensor_min_range, tensor_max_range, mode)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: x})
    return [res]
