#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_relu
"""
import tensorflow as tf

# pylint: disable=unused-argument,invalid-name
def relu_by_tf(x):
    """
    gather_v2_by_tf
    """
    x_holder = tf.placeholder(x.dtype, shape=x.shape)

    re = tf.nn.relu(x_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result


def calc_expect_func(x, y):
    """
    calc_expect_func
    """
    res = relu_by_tf(x["value"])
    return [res]

