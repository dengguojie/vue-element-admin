#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
from unicodedata import name
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops


def calc_expect_func(input_x, input_y, output_z, tolerance=1e-5):

    data_x = input_x.get('value')
    data_y = input_y.get('value')

    tensor_x = tf.placeholder(data_x.dtype, shape=data_x.shape)
    tensor_y = tf.placeholder(data_y.dtype, shape=data_y.shape)

    out = gen_math_ops.ApproximateEqual(x=tensor_x, y=tensor_y, tolerance=tolerance,name="ApproximateEqual")

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: data_x, tensor_y: data_y})
    return [res]