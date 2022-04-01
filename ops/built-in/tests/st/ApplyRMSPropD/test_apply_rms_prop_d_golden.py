#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_apply_adam_d_golden
"""
import tensorflow as tf
from tensorflow.python.training import gen_training_ops

def by_tf(var, ms, mom, grad, rho, momentum, epsilon):

    var_holder = tf.Variable(var)
    ms_holder = tf.Variable(ms)
    mom_holder = tf.Variable(mom)
    lr = 0.01
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)
    re = gen_training_ops.apply_rms_prop(var_holder, ms_holder, mom_holder, lr,
                                         rho, momentum, epsilon, grad_holder)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(re, feed_dict={grad_holder: grad})
    return res

def calc_expect_func(var, ms, mom, lr, grad,
                     var_out, ms_out, mom_out,
                     rho, momentum, epsilon):

    res =  by_tf(var["value"], ms["value"], mom["value"], grad["value"], rho, momentum, epsilon)
    return res

