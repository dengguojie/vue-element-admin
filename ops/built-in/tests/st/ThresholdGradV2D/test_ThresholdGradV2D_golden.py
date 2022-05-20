#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_threshold
"""
import tensorflow as tf

def golden(gradients, features, threshold):

    input_dtype = gradients.dtype
    gradients_tensor = tf.placeholder(gradients.dtype, shape=gradients.shape)
    features_tensor = tf.placeholder(features.dtype, shape=features.shape)
    threshold = tf.broadcast_to(tf.constant(threshold, dtype=input_dtype), gradients.shape)
    one_tensor = tf.broadcast_to(tf.constant(0, dtype=input_dtype), gradients.shape)

    features_less_threshold = tf.less(threshold, features_tensor)
    out = tf.where(features_less_threshold, gradients_tensor, one_tensor)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={gradients_tensor: gradients, features_tensor: features})
    return res

def calc_expect_func(input_gradients, input_features, output_backprops, threshold):

    gradients_value = input_gradients.get('value')
    features_value = input_features.get('value')
    res = golden(gradients_value, features_value, threshold)
    return [res]
