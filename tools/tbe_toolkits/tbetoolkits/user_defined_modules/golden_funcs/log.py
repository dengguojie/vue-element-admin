#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
log
"""
# 2022.2.16 wbl
import tbetoolkits
from .registry import register_golden


@register_golden(["log"])
def log(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    input_data_x1 = tf.compat.v1.placeholder(shape=context.input_arrays[0].shape,
                                          dtype=context.input_arrays[0].dtype)
    
    
    
    out = tf.log(input_data_x1)
    
    feed_dict = {input_data_x1: context.input_arrays[0]}
    
    init_op = tf.compat.v1.global_variables_initializer()  
    
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        out = sess.run(out, feed_dict=feed_dict)
    return out.astype(context.output_dtypes[0])
 
