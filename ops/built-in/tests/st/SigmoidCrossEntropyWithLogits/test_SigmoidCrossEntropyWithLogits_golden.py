#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def golden(predict_value, target_value):

    precision_dtype = predict_value.dtype
    predict = tf.placeholder(predict_value.dtype, shape=predict_value.shape)
    target = tf.placeholder(target_value.dtype, shape=target_value.shape)

    const_zero = tf.constant(0, dtype=precision_dtype)
    max_predict_zero = tf.maximum(predict, const_zero)
    abs_predict = tf.abs(predict)
    reverse_abs_predict = tf.subtract(const_zero, abs_predict)
    vexp_predict = tf.exp(reverse_abs_predict)
    const_one = tf.constant(1, dtype=precision_dtype)
    vadds_res = tf.add(vexp_predict, const_one)
    vlog_res = tf.log(vadds_res)
    vmul_res = tf.multiply(predict, target)
    res = tf.subtract(vlog_res, vmul_res)
    out = tf.add(res, max_predict_zero)

    feed_dict = {predict: predict_value, target: target_value}
    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return res


def calc_expect_func(predict, target, loss):

    predict_value = predict.get('value')
    target_value = target.get('value')

    res = golden(predict_value, target_value)

    return [res]
