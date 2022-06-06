#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.ops import gen_array_ops


def calc_expect_func(x, crops, y, block_size):

    data_x = x.get('value')
    data_crops = crops.get('value')
    tensor_x = tf.placeholder(data_x.dtype, shape=data_x.shape)

    out = gen_array_ops.batch_to_space(tensor_x, data_crops, block_size)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: data_x})
    return res