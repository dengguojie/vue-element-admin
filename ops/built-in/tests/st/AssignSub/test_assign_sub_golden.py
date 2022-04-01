#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_assign_sub
"""
import tensorflow as tf

def by_tf(var, value):

    var_holder = tf.Variable(var)
    value_holder = tf.placeholder(value.dtype, shape=value.shape)

    re = tf.assign_sub(var_holder, value_holder)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(re, feed_dict={value_holder: value})
    return res

def calc_expect_func(var, value, out):

    res =  by_tf(var["value"], value["value"])
    return [res]

