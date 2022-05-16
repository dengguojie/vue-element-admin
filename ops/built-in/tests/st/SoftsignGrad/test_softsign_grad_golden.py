#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
test_sotfsign
'''
# Third-Party Packages
import tensorflow as tf



def calc_expect_func(y_grad, x, x_grad):

    y_grad = y_grad.get('value')
    tensor_y_grad = tf.placeholder(y_grad.dtype, shape=y_grad.shape)

    x = x.get('value')
    tensor_x = tf.placeholder(x.dtype, shape=x.shape)
    out = tf.raw_ops.SoftsignGrad(gradients=y_grad, features=x, name="SoftsignGrad")

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_y_grad: y_grad, tensor_x: x})
    return [res]
