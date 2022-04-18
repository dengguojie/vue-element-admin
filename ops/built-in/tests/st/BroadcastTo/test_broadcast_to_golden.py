#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(x, shape, y):
    x = x.get('value')
    shape = shape.get('value')

    x_shape = x.shape
    shape_shape = shape.shape

    x_dtype = x.dtype
    shape_dtype = shape.dtype
 
    tensor_x = tf.placeholder(x_dtype, shape=x_shape)
    tensor_shape = tf.placeholder(shape_dtype, shape=shape_shape)

    feed_dict = {tensor_x: x, tensor_shape: shape}

    out = tf.broadcast_to(tensor_x, tensor_shape)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return [res]
