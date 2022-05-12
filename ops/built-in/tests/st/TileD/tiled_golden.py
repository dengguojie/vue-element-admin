#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def calc_expect_func(x, y, multiples):
    x = x.get('value')
    x_shape = x.shape
    x_dtype = x.dtype
    multiples_holder = tf.constant(multiples)
 
    tensor_x = tf.placeholder(x_dtype, shape=x_shape)
    feed_dict = {tensor_x: x}
    out = tf.tile(tensor_x, multiples_holder)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return [res]