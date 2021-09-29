#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.ops import inplace_ops


def calc_expect_func(x, indices, v, y):
    data = inplace_ops.inplace_update(x["value"], indices["value"], v["value"])
    with tf.compat.v1.Session() as sess:
        res = sess.run(data)
    return [res,]
