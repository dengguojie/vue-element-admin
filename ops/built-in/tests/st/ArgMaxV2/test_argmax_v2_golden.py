#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def calc_expect_func(x, dimension, y):

    x = x.get('value')
    dimension = dimension.get('value')[0]

    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    dimension_holder = tf.constant(dimension)

    out =  tf.argmax(x_holder, dimension_holder)
    out_int32 = tf.cast(out, tf.int32)

    with tf.Session() as sess:
        res = sess.run(out_int32, feed_dict={x_holder: x})
    return [res]
