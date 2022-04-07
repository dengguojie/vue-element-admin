import numpy as np
import tensorflow as tf


def reduce_by_tf(y, dy):
    y_holder = tf.compat.v1.placeholder(y.dtype, shape=y.shape)
    dy_holder = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
    re = tf.raw_ops.RsqrtGrad(y=y_holder, dy=dy_holder)
    with tf.compat.v1.Session() as sess:
        result = sess.run(re, feed_dict={y_holder: y, dy_holder: dy})
    return result


def calc_expect_func(input_y, input_dy, output_z):
    res = reduce_by_tf(input_y["value"], input_dy["value"])
    return [res]

