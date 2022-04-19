#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def calc_expect_func(var, accum, lr, grad, var_out, accum_out, update_slots):

    var = var.get('value')
    accum = accum.get('value')
    lr = lr.get('value')[0]
    grad = grad.get('value')

    var_holder = tf.Variable(var)
    accum_holder = tf.Variable(accum)
    lr_holder = tf.constant(lr)
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)

    out = gen_training_ops.apply_adagrad(var_holder, accum_holder, lr_holder, grad_holder, True, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={grad_holder: grad})
    return [res]
