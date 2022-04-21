#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(x, block_shape, crops, y):

    x = x.get('value')
    block_shape = block_shape.get('value')
    crops = crops.get('value')

    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    block_shape_holder = tf.placeholder(block_shape.dtype, shape=block_shape.shape)
    crops_holder = tf.placeholder(crops.dtype, shape=crops.shape)

    out = tf.batch_to_space_nd(x_holder, block_shape_holder, crops_holder)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={x_holder: x, block_shape_holder: block_shape, crops_holder: crops})
    return [res]
