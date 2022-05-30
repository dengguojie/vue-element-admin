#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.training import gen_training_ops


def calc_expect_func(var, accum, lr, grad, out_var, out_accum, epsilon, update_slots=True):

    data_var = var.get('value')
    data_accum = accum.get('value')
    data_lr = lr.get('value')[0]
    data_grad = grad.get('value')

    tensor_var = tf.Variable(data_var)
    tensor_accum = tf.Variable(data_accum)
    tensor_grad = tf.placeholder(data_grad.dtype, shape=data_grad.shape)

    s = gen_training_ops.apply_adagrad_v2(tensor_var,tensor_accum,data_lr,epsilon,tensor_grad,False,update_slots)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(s, feed_dict={tensor_grad: data_grad})
    return [res]