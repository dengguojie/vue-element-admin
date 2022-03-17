#!/usr/bin/env python3
# _*_ coding: UTF-8 _*_
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(x, bias, y, data_format):
    x = x.get('value')
    bias = bias.get('value')

    x_shape = x.shape
    bias_shape = bias.shape

    x_dtype = x.dtype
    bias_dtype = bias.dtype

    tensor_x = tf.placeholder(x_dtype, shape=x_shape)
    tensor_bias = tf.placeholder(bias_dtype, shape=bias_shape)

    feed_dict = {tensor_x:x, tensor_bias:bias}

    out = tf.nn.bias_add(tensor_x, tensor_bias, data_format)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return [res]
