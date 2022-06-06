#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import tensorflow as tf
import random


def calc_expect_func(x, y, axis):

    data_x = x.get('value')
    tensor_x = tf.placeholder(data_x.dtype, shape=data_x.shape)

    out = tf.layers.flatten(tensor_x)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: data_x})
    return res