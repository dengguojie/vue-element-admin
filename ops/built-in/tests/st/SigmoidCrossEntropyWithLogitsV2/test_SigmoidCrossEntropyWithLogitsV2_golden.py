#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def golden(predict_value, target_value, weight_value, pos_weight_value, reduction):

    precision_dtype = predict_value.dtype
    predict = tf.placeholder(predict_value.dtype, shape=predict_value.shape)
    target = tf.placeholder(target_value.dtype, shape=target_value.shape)
    weight = tf.placeholder(weight_value.dtype, shape=weight_value.shape)
    pos_weight = tf.placeholder(pos_weight_value.dtype, shape=pos_weight_value.shape)

    const_zero = tf.constant(0, dtype=precision_dtype)
    const_one = tf.constant(1, dtype=precision_dtype)

    reversed_predict = tf.subtract(const_zero, predict)
    max_predict_zero = tf.maximum(reversed_predict, const_zero)

    reversed_max_predict_zero = tf.subtract(const_zero, max_predict_zero)
    exp_reversed_max_predict_zero = tf.exp(reversed_max_predict_zero)
    sub_reversed_max_predict_zero = tf.subtract(reversed_max_predict_zero, predict)
    exp_sub_reversed_max_predict_zero = tf.exp(sub_reversed_max_predict_zero)
    add_reversed_predict = tf.add(exp_reversed_max_predict_zero, exp_sub_reversed_max_predict_zero)
    log_reversed_predict = tf.log(add_reversed_predict)
    add_max_predict = tf.add(log_reversed_predict, max_predict_zero)

    sub_target = tf.subtract(const_one, target)
    mul_predict_target = tf.multiply(sub_target, predict)

    sub_pos_weight = tf.subtract(pos_weight, const_one)
    mul_pos_weight = tf.multiply(sub_pos_weight, target)
    add_pos_weight = tf.add(mul_pos_weight, const_one)
    mul_pos_weight_predict = tf.multiply(add_pos_weight, add_max_predict)
    loss = tf.add(mul_predict_target, mul_pos_weight_predict)
    out = tf.multiply(loss, weight)

    feed_dict = {
        predict: predict_value, target: target_value,
        weight: weight_value, pos_weight: pos_weight_value
        }
    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return res


def calc_expect_func(predict, target, weight, pos_weight, loss, reduction):

    predict_value = predict.get('value')
    target_value = target.get('value')
    weight_value = weight.get('value')
    pos_weight_value = pos_weight.get('value')

    res = golden(predict_value, target_value, weight_value, pos_weight_value, reduction)

    return [res]
