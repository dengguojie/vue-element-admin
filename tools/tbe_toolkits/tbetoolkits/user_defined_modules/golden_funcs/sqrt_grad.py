#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
sqrt_grad
"""
# 2022.2.17 zwx

import tbetoolkits
from .registry import register_golden
from tensorflow.python.ops import gen_nn_ops

@register_golden(["sqrt_grad"])
def sigmsqrt_gradoid_grad(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    input_data = tf.compat.v1.placeholder(shape=context.input_arrays[0].shape,
                                          dtype=context.input_arrays[0].dtype)
    input_data1 = tf.compat.v1.placeholder(shape=context.input_arrays[1].shape,
                                          dtype=context.input_arrays[1].dtype)

    out = gen_nn_ops.sqrt_grad(input_data, input_data1)


    feed_dict = {input_data: context.input_arrays[0], input_data1: context.input_arrays[1]}

    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        out = sess.run(out, feed_dict=feed_dict)
    return out.astype(context.output_dtypes[0])
