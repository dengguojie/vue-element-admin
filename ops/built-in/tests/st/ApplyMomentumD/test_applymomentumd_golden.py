#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_apply_adam_d_golden
"""
import tensorflow as tf
from tensorflow.python.training import gen_training_ops

def by_tf(var, accum, grad, use_locking, use_nesterov):

    var_holder = tf.Variable(var)
    accum_holder = tf.Variable(accum)
    lr = 0.5
    momentum = 0.5
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)

    re = gen_training_ops.apply_momentum(var_holder, accum_holder, lr,
                                         grad_holder, momentum, use_locking, use_nesterov)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(re, feed_dict={grad_holder: grad})
    return res

def calc_expect_func(var, accum, lr, grad, momentum,
                     var_out, accum_out, use_locking, use_nesterov):

    res =  by_tf(var["value"], accum["value"], grad["value"], use_locking, use_nesterov)
    return res

