#!/usr/bin/env python3
# _*_ coding: UTF-8 _*_
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


# pylint: disable=unused-argument, invalid-name
def reduce_max_by_tf(x1, x2):
    """
    div_by_tf
    """
    x1_holder = tf.placeholder(x1.dtype, shape=x1.shape)
    x2_holder = tf.placeholder(x2.dtype, shape=x2.shape)
    re = tf.reduce_max(x1_holder, x2_holder)

    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x1_holder:x1, x2_holder:x2})
    return result

def calc_expect_func(x, axes, y, keep_dims):
    """
    calc_expect_func
    """
    res = reduce_max_by_tf(x["value"], axes["value"])
    return [res]
