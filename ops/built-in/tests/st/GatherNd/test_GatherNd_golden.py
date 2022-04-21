#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def calc_expect_func(dict_data, dict_indices, dict_out):

    dict_data = dict_data.get('value')
    dict_indices = dict_indices.get('value')

    dict_data_holder = tf.placeholder(dict_data.dtype, shape=dict_data.shape)
    dict_indices_holder = tf.placeholder(dict_indices.dtype, shape=dict_indices.shape)

    out = tf.gather_nd(dict_data_holder, dict_indices_holder)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={dict_data_holder: dict_data, dict_indices_holder: dict_indices})
    return [res]
