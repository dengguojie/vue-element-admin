#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.ops import math_grad


def calc_expect_func(y, dy, z):

    x = y.get('value')
    y = dy.get('value')

    tensor_x = tf.placeholder(x.dtype, shape=x.shape)
    tensor_y = tf.placeholder(y.dtype, shape=y.shape)

    b1 = tf.abs(tensor_x)
    out = math_grad._AbsGrad(b1.op, tensor_y)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: x, tensor_y: y})
    return [res]
