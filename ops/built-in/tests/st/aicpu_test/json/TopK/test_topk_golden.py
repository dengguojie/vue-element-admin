import os
import numpy as np
import tensorflow as tf

def get_tf_type(dtype):
    if dtype == "int8":
        return tf.int8
    elif dtype == "int16":
        return tf.int16
    elif dtype == "int32":
        return tf.int32
    elif dtype == "int64":
        return tf.int64
    elif dtype == "uint8":
        return tf.uint8
    elif dtype == "uint16":
        return tf.uint16
    elif dtype == "uint32":
        return tf.uint32
    elif dtype == "uint64":
        return tf.uint64
    elif dtype == "float":
        return tf.float32
    elif dtype == "double":
        return tf.float64
    elif dtype == "float16":
        return tf.float16

def run_tf_top_k(x_data, x_shape, x_dtype, k_data, sorted):
    x1 = tf.compat.v1.placeholder(x_dtype, shape=x_shape)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=[])
    re = tf.math.top_k(x1, x2, sorted)
    with tf.compat.v1.Session() as session:
        data = session.run(re, feed_dict={x1:x_data, x2:k_data})
    return (data[0], data[1])

def calc_expect_func(input_x, input_k, output_values, output_indices, attr_sorted):
    x_type = get_tf_type(input_x["dtype"])
    res1, res2 = run_tf_top_k(input_x["value"], input_x["shape"], x_type, input_k["value"][0], attr_sorted)
    return [res1, res2]
