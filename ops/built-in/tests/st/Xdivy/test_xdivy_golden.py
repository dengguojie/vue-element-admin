#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(x1, x2, y):

    x1 = x1.get('value')
    x2 = x2.get('value')

    tensor_x1 = tf.placeholder(x1.dtype, shape=x1.shape)
    tensor_x2 = tf.placeholder(x2.dtype, shape=x2.shape)

    out = tf.math.xdivy(tensor_x1, tensor_x2)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x1: x1, tensor_x2: x2})
    return [res]
