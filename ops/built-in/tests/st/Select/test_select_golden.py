#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(condition, input_x, input_y, output_z):
    condition = condition.get('value')
    input_x = input_x.get('value')
    input_y =input_y.get('value')

    con_shape = condition.shape
    x_shape = input_x.shape
    y_shape = input_y.shape

    con_dtype = condition.dtype
    x_dtype = input_x.dtype
    y_dtype = input_y.dtype
 

    tensor_con = tf.placeholder(con_dtype, shape=con_shape)
    tensor_con = tf.cast(tensor_con, tf.bool)
    tensor_x = tf.placeholder(x_dtype, shape=x_shape)
    tensor_y = tf.placeholder(y_dtype, shape=y_shape)

    feed_dict = {tensor_con: condition, tensor_x: input_x, tensor_y: input_y}

    out = tf.where(tensor_con, tensor_x, tensor_y)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return [res]