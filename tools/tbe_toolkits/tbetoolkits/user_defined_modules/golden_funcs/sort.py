#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for sort
"""
# Third-Party Packages
import tbetoolkits
from .registry import register_golden


@register_golden(["top_k_d"])
def _transpose_dsl(context: tbetoolkits.UniversalTestcaseStructure):
    import tensorflow as tf
    input_data = tf.compat.v1.placeholder(shape=context.input_arrays[0].shape,
                                          dtype=context.input_arrays[0].dtype)
    out = tf.math.top_k(input_data, k=context.other_compilation_params["k"],
                        sorted=context.other_compilation_params["sorted"])
    feed_dict = {input_data: context.input_arrays[0]}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        out_k, out_indices = sess.run(out, feed_dict=feed_dict)
    return out_k.astype(context.output_dtypes[0]), out_indices.astype(context.output_dtypes[1])
