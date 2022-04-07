import numpy as np
import tensorflow as tf


def reduce_by_tf(x):
    x_holder = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    re = tf.raw_ops.Sigmoid(x=x_holder)
    with tf.compat.v1.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result


def calc_expect_func(x, y):
    res = reduce_by_tf(x["value"])
    return [res]

