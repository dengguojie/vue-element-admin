#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.ops import math_grad


def calc_expect_func(y, dy, z):

    data_y = y.get('value')
    data_dy = dy.get('value')

    tensor_x = tf.placeholder(data_y.dtype, shape=data_y.shape)
    tensor_y = tf.placeholder(data_dy.dtype, shape=data_dy.shape)

    out = -1 * data_dy / (tf.math.sqrt(1-tf.math.square(data_y)))

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: data_y, tensor_y: data_dy})
    return [res]