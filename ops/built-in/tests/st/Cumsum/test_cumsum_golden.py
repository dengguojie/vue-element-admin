#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def calc_expect_func(x, axis, y, exclusive, reverse):
  
    x_value = x.get("value")
    axis_value = axis.get("value")[0]
    x_holder = tf.placeholder(dtype=x_value.dtype, shape=x_value.shape)

    y = tf.cumsum(x_holder, axis=axis_value, exclusive=exclusive, reverse=reverse)
   
    with tf.Session() as sess:
        result = sess.run(y, feed_dict={x_holder: x_value})
    return [result]
