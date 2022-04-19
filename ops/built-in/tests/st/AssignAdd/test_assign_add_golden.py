#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_assign_sub
"""
import tensorflow as tf

def by_tf(ref, value):

    ref_holder = tf.Variable(ref)
    value_holder = tf.placeholder(value.dtype, shape=value.shape)

    re = tf.assign_add(ref_holder, value_holder)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(re, feed_dict={value_holder: value})
    return res

def calc_expect_func(ref, value, out):

    res =  by_tf(ref["value"], value["value"])
    return [res]
