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

def calc_expect_func(input_y,input_dy,output_data):
    y = input_y.get('value')
    dy = input_dy.get('value')
    tensor_y = tf.placeholder(y.dtype, shape=y.shape)
    tensor_dy = tf.placeholder(dy.dtype, shape=dy.shape)

    out = tf.raw_ops.ReciprocalGrad(y=tensor_y, dy=tensor_dy, name="ReciprocalGrad")

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_y:y, tensor_dy:dy})
    return [res]
