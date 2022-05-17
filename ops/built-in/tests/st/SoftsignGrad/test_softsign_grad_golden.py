#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
test_sotfsign
'''
# Third-Party Packages
import tensorflow as tf



def calc_expect_func(gradients, features, output):

    gradients = gradients.get('value')
    tensor_gradients = tf.placeholder(gradients.dtype, shape=gradients.shape)

    features = features.get('value')
    tensor_features = tf.placeholder(features.dtype, shape=features.shape)
    out = tf.raw_ops.SoftsignGrad(gradients=gradients, features=features, name="SoftsignGrad")

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_gradients: gradients, tensor_features: features})
    return [res]
