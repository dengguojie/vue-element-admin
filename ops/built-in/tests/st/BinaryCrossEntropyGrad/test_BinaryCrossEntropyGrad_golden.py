#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_square_sum_v1
"""
import tensorflow as tf

def golden(x, y, grad_output, weight, reduction):

    cof = 1.0
    if reduction == "mean":
        reduce_elts = 1.0
        for i in x.shape:
            reduce_elts *= i
        cof = reduce_elts ** (-1)

    x_tensor = tf.placeholder(x.dtype, shape=x.shape)
    y_tensor = tf.placeholder(y.dtype, shape=y.shape)
    grad_output_tensor = tf.placeholder(grad_output.dtype, shape=grad_output.shape)
    weight_tensor = tf.placeholder(weight.dtype, shape=weight.shape)

    val1 = tf.subtract(x_tensor, y_tensor)
    minus_predict = tf.multiply(x_tensor, -1)
    val2_tmp = tf.add(minus_predict, 1)
    val2 = tf.multiply(x_tensor, val2_tmp)
    val2 = tf.maximum(val2, 1e-12)
    result = tf.divide(val1, val2)
    result = tf.multiply(weight_tensor, result)
    result = tf.multiply(grad_output_tensor, result)
    out = tf.multiply(result, cof)

    feed_dict = {x_tensor: x, y_tensor: y, grad_output_tensor: grad_output, weight_tensor: weight}
    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return res

def calc_expect_func(x, y, grad_output, weight, output, reduction):

    x = x.get('value')
    y = y.get('value')
    grad_output = grad_output.get('value')
    weight = weight.get('value')
    res = golden(x, y, grad_output, weight, reduction)

    return [res]
