#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.ops import math_grad


def calc_expect_func(x,y,value):

    x = x.get('value')

    tensor_x = tf.placeholder(x.dtype, shape=x.shape)

    value_val = tf.constant(value,dtype=x.dtype)
    out = tf.add(x,value_val,name="adds")

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: x})
    return [res]