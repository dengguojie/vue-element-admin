import numpy as np
import tensorflow as tf


def reduce_by_tf(x, axes, keep_dims):
    x_holder = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    re = tf.reduce_prod(x_holder, axes, keep_dims)
    with tf.compat.v1.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result


def calc_expect_func(x, axes, y, keep_dims):
    res = reduce_by_tf(x["value"], axes["value"], keep_dims)
    return [res]

