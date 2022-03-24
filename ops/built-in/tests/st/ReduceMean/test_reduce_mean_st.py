import numpy as np
import tensorflow as tf

def reduce_mean_tf(x, axes, keepdims):
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.reduce_mean(x_holder, axes, keepdims)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result

def reduce_mean(x, y, axes, keep_dims=None, kernel_name="reduce_mean", impl_mode=None):
    input_x_data = x['value']
    axis = axes['value']
    res = reduce_mean_tf(input_x_data, axis, keep_dims)
    return res