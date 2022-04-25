#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.ops import array_ops

def calc_expect_func(shape, begin, end, strides, dy, output,
                     begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):

    x = shape.get('value')
    begin = begin.get('value')
    end = end.get('value')
    strides = strides.get('value')
    dy = dy.get('value')

    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    begin_holder = tf.placeholder(begin.dtype, shape=begin.shape)
    end_holder = tf.placeholder(end.dtype, shape=end.shape)
    strides_holder = tf.placeholder(strides.dtype, shape=strides.shape)
    dy_holder = tf.placeholder(dy.dtype, shape=dy.shape)

    out =  array_ops.strided_slice_grad(x_holder, begin_holder, end_holder, strides_holder, dy_holder,
                                        begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={x_holder: x, begin_holder: begin, end_holder: end,
                                       strides_holder: strides, dy_holder: dy})
    return [res]
