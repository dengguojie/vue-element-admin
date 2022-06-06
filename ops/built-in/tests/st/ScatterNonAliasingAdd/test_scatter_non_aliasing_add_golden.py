#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_array_ops


def calc_expect_func(var, indices, adds, var_out):

    data_var = var.get('value')
    data_indices = indices.get('value')
    data_adds = adds.get('value')
    tensor_var = tf.Variable(data_var)
    tensor_indices = tf.placeholder(data_indices.dtype, shape=data_indices.shape)
    tensor_adds = tf.placeholder(data_adds.dtype, shape=data_adds.shape)

    out = gen_array_ops.scatter_nd_non_aliasing_add(tensor_var,tensor_indices, tensor_adds)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={tensor_indices: data_indices, tensor_adds: data_adds})
    return res