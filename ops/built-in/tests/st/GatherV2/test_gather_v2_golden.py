#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_gather_v2
"""
import tensorflow as tf
from tensorflow.python.ops import array_ops


# pylint: disable=unused-argument,invalid-name
def gather_by_tf(x, indices, axis, batch_dims):
    """
    gather_v2_by_tf
    """
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    indices_holder = tf.placeholder(indices.dtype, shape=indices.shape)
    axis_holder = tf.placeholder(axis.dtype, shape=axis.shape)

    re = array_ops.gather_v2(x_holder, indices_holder, axis_holder, batch_dims=batch_dims)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x, indices_holder: indices, axis_holder: axis})
    return result


def calc_expect_func(x, indices, axis, y, batch_dims):
    """
    calc_expect_func
    """
    res = gather_by_tf(x["value"], indices["value"], axis["value"], batch_dims)
    return [res]

