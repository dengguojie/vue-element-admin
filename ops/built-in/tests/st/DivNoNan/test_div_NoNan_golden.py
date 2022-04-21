#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_div_NoNan
"""
import tensorflow as tf

def by_tf(x1, x2):

    x1_holder = tf.placeholder(x1.dtype, shape=x1.shape)
    x2_holder = tf.placeholder(x2.dtype, shape=x2.shape)

    re = tf.math.divide_no_nan(x1_holder, x2_holder)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(re, feed_dict={x1_holder: x1, x2_holder: x2})
    return res

def calc_expect_func(x1, x2, y):

    res =  by_tf(x1["value"], x2["value"])
    return [res]
