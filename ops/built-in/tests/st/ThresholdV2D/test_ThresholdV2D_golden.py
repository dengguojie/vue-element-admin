#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_threshold
"""
import tensorflow as tf

def golden(x, threshold, value):

    input_dtype = x.dtype
    input_x = tf.placeholder(x.dtype, shape=x.shape)
    threshold = tf.broadcast_to(tf.constant(threshold, dtype=input_dtype), x.shape)
    value = tf.broadcast_to(tf.constant(value, dtype=input_dtype), x.shape)

    input_x_less_threshold = tf.less(threshold, input_x)
    out = tf.where(input_x_less_threshold, input_x, value)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={input_x: x})
    return res

def calc_expect_func(x, y, threshold, value):

    x_value = x.get('value')
    res = golden(x_value, threshold, value)

    return [res]
