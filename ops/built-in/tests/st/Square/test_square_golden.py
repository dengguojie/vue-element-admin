#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_square_v2
"""
import tensorflow as tf

# pylint: disable=unused-argument,invalid-name
def by_tf(x):
    """
    gather_v2_by_tf
    """
    x_holder = tf.placeholder(x.dtype, shape=x.shape)

    re = tf.square(x_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result


def calc_expect_func(input_x, output_y):
    """
    calc_expect_func
    """
    res = by_tf(input_x["value"])
    return [res]

