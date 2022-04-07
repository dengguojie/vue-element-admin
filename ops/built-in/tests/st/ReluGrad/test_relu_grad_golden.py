import numpy as np
import tensorflow as tf


def reduce_by_tf(gradients, features):
    gradients_holder = tf.compat.v1.placeholder(gradients.dtype, shape=gradients.shape)
    features_holder = tf.compat.v1.placeholder(features.dtype, shape=features.shape)
    re = tf.raw_ops.ReluGrad(gradients=gradients_holder, features=features_holder)
    with tf.compat.v1.Session() as sess:
        result = sess.run(re, feed_dict={gradients_holder: gradients, features_holder: features})
    return result


def calc_expect_func(input_gradients, input_features, output_backprops):
    res = reduce_by_tf(input_gradients["value"], input_features["value"])
    return [res]

