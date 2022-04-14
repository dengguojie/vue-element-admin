#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_assign
"""
import tensorflow as tf

def by_tf(x):

    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.nn.elu(x_holder)
    with tf.Session() as sess:
        res = sess.run(re, feed_dict={x_holder: x})
    return res

def calc_expect_func(x, y, alpha):

    res =  by_tf(x["value"])
    return [res]

