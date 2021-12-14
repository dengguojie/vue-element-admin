#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf


def calc_expect_func(x, segment_ids, y):
    data = tf.math.segment_sum(x["value"], tf.constant(segment_ids["value"]))
    with tf.Session() as sess:
        res = sess.run(data)
    return [res,]
