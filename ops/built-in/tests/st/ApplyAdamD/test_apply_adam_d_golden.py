#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_apply_adam_d_golden
"""
import tensorflow as tf
from tensorflow.python.training import gen_training_ops

def by_tf(var, m, v, grad, use_locking, use_nesterov):

    var_holder = tf.Variable(var)
    m_holder = tf.Variable(m)
    v_holder = tf.Variable(v)
    beta1_power = 0.5
    beta2_power = 0.5
    lr = 0.5
    beta1 = 0.5
    beta2 = 0.5
    epsilon = 0.5
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)
    re = gen_training_ops.apply_adam(var_holder, m_holder, v_holder, beta1_power, beta2_power,lr,
                                     beta1, beta2, epsilon, grad_holder, use_locking, use_nesterov)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(re, feed_dict={grad_holder: grad})
    return res

def calc_expect_func(var, m, v, beta1_power, beta2_power,
                    lr, beta1, beta2, epsilon, grad,
                    var_out, m_out, v_out, use_locking, use_nesterov):

    res =  by_tf(var["value"], m["value"], v["value"], grad["value"], use_locking, use_nesterov)
    return res

