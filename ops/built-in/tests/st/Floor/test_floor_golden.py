#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def calc_expect_func(x, y):

    x = x.get('value')
    x_holder = tf.placeholder(x.dtype, shape=x.shape)

    out = tf.floor(x_holder)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={x_holder: x})
    return [res]
