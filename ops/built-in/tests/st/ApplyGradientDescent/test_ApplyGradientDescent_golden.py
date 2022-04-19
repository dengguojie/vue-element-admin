#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def calc_expect_func(var, alpha, delta, out):

    var = var.get('value')
    alpha = alpha.get('value')[0]
    delta = delta.get('value')

    var_holder = tf.Variable(var)
    alpha_holder = tf.constant(alpha)
    delta_holder = tf.placeholder(delta.dtype, shape=delta.shape)

    out = gen_training_ops.apply_gradient_descent(var_holder, alpha_holder, delta_holder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={delta_holder: delta})
    return [res]
