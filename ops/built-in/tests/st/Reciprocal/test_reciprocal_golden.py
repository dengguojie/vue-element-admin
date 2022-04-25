#!/usr/bin/env python3
# _*_ coding: UTF-8 _*_
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


# pylint: disable=unused-argument, invalid-name
def reciprocal_by_tf(x1):
    """
    div_by_tf
    """
    x1_holder = tf.placeholder(x1.dtype, shape=x1.shape)
    re = tf.reciprocal(x1_holder)

    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x1_holder:x1})
    return result

def calc_expect_func(input_x, output_y):
    """
    calc_expect_func
    """
    res = reciprocal_by_tf(input_x["value"])
    return [res]
