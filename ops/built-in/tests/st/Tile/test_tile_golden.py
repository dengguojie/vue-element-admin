#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(x, multiples, y):
    x = x.get('value')
    multiples = multiples.get('value')

    x_shape = x.shape
    multiples_shape = multiples.shape

    x_dtype = x.dtype
    multiples_dtype = multiples.dtype
 
    tensor_x = tf.placeholder(x_dtype, shape=x_shape)
    tensor_multiples = tf.placeholder(multiples_dtype, shape=multiples_shape)

    feed_dict = {tensor_x: x, tensor_multiples: multiples}

    out = tf.tile(tensor_x, tensor_multiples)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return [res]