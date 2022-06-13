#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(input_x, input_a, output_y):
    input_x = input_x.get('value')
    input_a =input_a.get('value')

    x_shape = input_x.shape
    a_shape = input_a.shape

    x_dtype = input_x.dtype
    a_dtype = input_a.dtype
 
    tensor_x = tf.placeholder(x_dtype, shape=x_shape)
    tensor_a = tf.placeholder(a_dtype, shape=a_dtype)

    feed_dict = {tensor_x: input_x, tensor_a: input_a}
    pos = tf.nn.relue(input_x)
    neg = input_a * (input_x - abs(input_x)) * 0.5
    out = pos + neg

    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return [res]