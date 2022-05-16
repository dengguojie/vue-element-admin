#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(input_x, input_y, output_x):

    x = input_x.get('value')
    y = input_y.get('value')

    tensor_x = tf.placeholder(x.dtype, shape=x.shape)
    tensor_y = tf.placeholder(y.dtype, shape=y.shape)

    out = tf.truncatemod(tensor_x, tensor_y)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: x, tensor_y: y})
    return [res]
