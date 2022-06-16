#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_square_sum_v1
"""
import tensorflow as tf

def golden(x, axis, keep_dims):

    x_tensor = tf.placeholder(x.dtype, shape=x.shape)
    x_square = tf.square(x_tensor)
    out = tf.reduce_sum(x_square, axis, keep_dims)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={x_tensor: x})
    return res

def calc_expect_func(input_x, output1, axis, keep_dims):

    x = input_x.get('value')
    res = golden(x, axis, keep_dims)

    return [res]

def calc_fnz_expect_func(input_x, output1, axis, keep_dims):

    x = input_x.get('value')
    axis = [0, 1, 2, 3]
    res = golden(x, axis, keep_dims)

    return [res]
