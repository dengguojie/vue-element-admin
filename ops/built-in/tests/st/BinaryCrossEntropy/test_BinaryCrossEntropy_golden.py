#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_binary_cross_entropy_golden.py
"""
import tensorflow as tf

def golden(x, y, weight, reduction):

    x_tensor = tf.placeholder(x.dtype, shape=x.shape)
    y_tensor = tf.placeholder(y.dtype, shape=y.shape)
    weight_tensor = tf.placeholder(weight.dtype, shape=weight.shape)
    axis = [i for i in range(len(x.shape))]

    # calcu value : y * log(x)
    x_max = tf.maximum(x_tensor, 1e-12)
    x_log_tmp = tf.log(x_max)
    data_mul1 = tf.multiply(x_log_tmp, y_tensor)

    # calcu value : (1-y) * log(1-x)
    x_neg_tmp = tf.multiply(x_max, -1.0)
    x1_tmp = tf.add(x_neg_tmp, 1.0)
    y_neg_tmp = tf.multiply(y_tensor, -1.0)
    y1_tmp = tf.add(y_neg_tmp, 1.0)
    x1_tmp = tf.maximum(x1_tmp, 1e-12)
    x1_log_tmp = tf.log(x1_tmp)
    data_mul2 = tf.multiply(x1_log_tmp, y1_tmp)

    # calcu value : y * log(x) + (1-y) * log(1-x)
    data_sum = tf.add(data_mul1, data_mul2)
    # calcu value : -(y * log(x) + (1-y) * log(1-x))
    result = tf.multiply(data_sum, -1.0)
    result = tf.multiply(result, weight)
    if reduction == "mean":
        reduce_elts = 1.0
        for i in x.shape:
            reduce_elts *= i
        cof = reduce_elts ** (-1)
        result = tf.multiply(result, cof)
        out = tf.reduce_sum(result, axis=axis, keepdims=False)
    elif reduction == "sum":
        out = tf.reduce_sum(result, axis=axis, keepdims=False)
    elif reduction == "none":
        out = result

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={x_tensor: x, y_tensor: y, weight_tensor: weight})
    return res

def calc_expect_func(x, y, weight, output, reduction):
    input_x = x.get('value')
    input_y = y.get('value')
    input_weight = weight.get('value')
    res = golden(input_x, input_y, input_weight, reduction)

    return [res]
