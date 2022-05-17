#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def golden(predict_value, target_value, dout_value, weight_value, pos_weight_value, reduction):

    precision_dtype = predict_value.dtype
    predict = tf.placeholder(predict_value.dtype, shape=predict_value.shape)
    target = tf.placeholder(target_value.dtype, shape=target_value.shape)
    dout = tf.placeholder(dout_value.dtype, shape=dout_value.shape)
    weight = tf.placeholder(weight_value.dtype, shape=weight_value.shape)
    pos_weight = tf.placeholder(pos_weight_value.dtype, shape=pos_weight_value.shape)

    exp_predict = tf.exp(predict)
    exp_add1 = tf.add(exp_predict, tf.constant(1, precision_dtype))
    sigmoid_tmp = tf.div(exp_predict, exp_add1)
    sigmoid_res = tf.cast(sigmoid_tmp, precision_dtype)

    log_weight = tf.multiply(pos_weight, target)
    weight_tmp = tf.add(log_weight, tf.constant(1, precision_dtype))
    weight_sub = tf.subtract(weight_tmp, target)
    grad_tmp = tf.multiply(weight_sub, sigmoid_res)
    grad_cur = tf.subtract(grad_tmp, log_weight)
    grad_output = tf.multiply(grad_cur, dout)

    out = tf.multiply(grad_output, weight)

    feed_dict = {
        predict: predict_value, target: target_value,
        dout: dout_value, weight: weight_value,
        pos_weight: pos_weight_value
        }
    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return res


def calc_expect_func(predict, target, dout, weight, pos_weight, gradient, reduction):

    predict_value = predict.get('value')
    target_value = target.get('value')
    dout_value = dout.get('value')
    weight_value = weight.get('value')
    pos_weight_value = pos_weight.get('value')

    res = golden(predict_value, target_value, dout_value, weight_value, pos_weight_value, reduction)

    return [res]
