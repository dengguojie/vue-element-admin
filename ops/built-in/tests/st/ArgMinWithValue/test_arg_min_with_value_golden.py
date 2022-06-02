#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
from unicodedata import name
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops


def calc_expect_func(x, index, value, dimension, keep_dims):

    data_x = x.get('value')
    tensor_x = tf.placeholder(data_x.dtype, shape=data_x.shape)

    out = tf.argmin(tensor_x, dimension, output_type=tf.int32)
    value = tf.reduce_min(tensor_x, dimension,keep_dims=False, name='reducdemin')

    with tf.Session() as sess:
        res1 = sess.run(out, feed_dict={tensor_x: data_x})
        res2 = sess.run(value, feed_dict={tensor_x: data_x})
    return res1, res2