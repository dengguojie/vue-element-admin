#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_add_golden
"""

'''
Special golden data generation function for ops add
'''

#Third-Party Packages
import tensorflow as tf

def calc_expect_func(x, dimension, y):
    x = x.get('value')
    dimension = dimension.get('value')[0]
    tensor_x1 = tf.placeholder(x.dtype, shape=x.shape)
    tensor_x2 = tf.constant(dimension)

    out = tf.argmin(tensor_x1, tensor_x2)
    out_int32 = tf.cast(out, tf.int32)

    with tf.Session() as sess:
        res = sess.run(out_int32, feed_dict={tensor_x1:x})
    return [res]
