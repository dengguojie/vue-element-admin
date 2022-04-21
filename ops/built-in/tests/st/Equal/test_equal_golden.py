#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def calc_expect_func(input_x, input_y, output_z):

    x = input_x.get('value')
    y = input_y.get('value')

    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    y_holder = tf.placeholder(y.dtype, shape=y.shape)

    out = tf.equal(x_holder, y_holder)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={x_holder: x, y_holder: y})
    return [res]
