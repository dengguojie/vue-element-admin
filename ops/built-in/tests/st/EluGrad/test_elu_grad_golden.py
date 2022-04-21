#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

def calc_expect_func(grads, activations, y):

    grads = grads.get('value')
    activations = activations.get('value')

    grads_holder = tf.placeholder(grads.dtype, shape=grads.shape)
    activations_holder = tf.placeholder(activations.dtype, shape=activations.shape)

    out = gen_nn_ops.elu_grad(grads_holder, activations_holder)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={grads_holder: grads, activations_holder: activations})
    return [res]
