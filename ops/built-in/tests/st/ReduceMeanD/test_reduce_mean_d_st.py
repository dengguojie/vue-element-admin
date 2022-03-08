import numpy as np
import tensorflow as tf

def reduce_mean_tf(x, axes, keepdims):
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.reduce_mean(x_holder, axes, keepdims)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result

def reduce_mean_d(x, y, axes, keep_dims=None, noop_with_empty_axes=True, kernel_name="reduce_mean_d", impl_mode=None):
    input_x_data = x['value']
    if len(axes) == 0 and not noop_with_empty_axes:
        axes = list(range(len(input_x_data.shape)))
    res = reduce_mean_tf(input_x_data, axes, keep_dims)
    return res