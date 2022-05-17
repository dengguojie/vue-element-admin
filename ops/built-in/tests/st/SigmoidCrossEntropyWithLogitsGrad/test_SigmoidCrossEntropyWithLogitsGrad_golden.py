#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf

def golden(predict_value, target_value, dout_value):

    precision_dtype = predict_value.dtype
    predict = tf.placeholder(predict_value.dtype, shape=predict_value.shape)
    target = tf.placeholder(target_value.dtype, shape=target_value.shape)
    dout = tf.placeholder(dout_value.dtype, shape=dout_value.shape)

    # e^x
    val1 = tf.exp(predict)
    # 1 + e^x
    val2 = tf.add(val1, tf.constant(1, dtype=precision_dtype))

    val3 = tf.div(val1, val2)
    # -target
    val4 = tf.multiply(target, tf.constant(-1, dtype=precision_dtype))

    val5 = tf.add(val3, val4)

    out = tf.multiply(val5, dout)

    feed_dict = {
        predict: predict_value, target: target_value, dout: dout_value
        }
    with tf.Session() as sess:
        res = sess.run(out, feed_dict=feed_dict)
    return res


def calc_expect_func(predict, target, dout, gradient):

    predict_value = predict.get('value')
    target_value = target.get('value')
    dout_value = dout.get('value')

    res = golden(predict_value, target_value, dout_value)

    return [res]
