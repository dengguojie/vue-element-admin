#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

def calc_expect_func(gradients, features, backprops, negative_slope):

    gradients = gradients.get('value')
    features = features.get('value')

    gradients_holder = tf.placeholder(gradients.dtype, shape=gradients.shape)
    features_holder = tf.placeholder(features.dtype, shape=features.shape)

    out = gen_nn_ops.leaky_relu_grad(gradients_holder, features_holder, negative_slope)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={gradients_holder: gradients, features_holder: features})
    return [res]
