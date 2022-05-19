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

def calc_expect_func(x, y):
    x = x.get('value')
    tensor_x = tf.placeholder(x.dtype, shape=x.shape)

    out = tf.ones_like(tensor_x)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x:x})
    return [res]
