#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
relu6_d
"""
# 2022.2.17 zwx1143009

import tbetoolkits
from .registry import register_golden


@register_golden(["relu6_d"])
def relu6_d(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    input_data = tf.compat.v1.placeholder(shape=context.input_arrays[0].shape,
                                          dtype=context.input_arrays[0].dtype)

    out = tf.nn.relu6(input_data)

    feed_dict = {input_data: context.input_arrays[0]}

    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        out = sess.run(out, feed_dict=feed_dict)
    return out.astype(context.output_dtypes[0])
