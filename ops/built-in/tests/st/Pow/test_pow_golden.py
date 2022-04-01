#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_gather_v2
"""
import tensorflow as tf

# pylint: disable=unused-argument,invalid-name
def gather_by_tf(x, y):
    """
    gather_v2_by_tf
    """
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    y_holder = tf.placeholder(y.dtype, shape=y.shape)
  
    re = tf.pow(x_holder, y_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x, y_holder: y})
    return result


def calc_expect_func(input_x, input_y, output_z):
    """
    calc_expect_func
    """
    res = gather_by_tf(input_x["value"], input_y["value"])
    return [res]

