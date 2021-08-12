import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import gen_array_ops


def tf_mirror_pad(x, paddings, mode):
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    paddings_holder = tf.placeholder(paddings.dtype, shape=paddings.shape)
    re = gen_array_ops.mirror_pad(x_holder, paddings_holder, mode=mode)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x, paddings_holder: paddings})
    return result


def calc_expect_func(x, paddings, y, mode):
    res = tf_mirror_pad(x["value"], paddings["value"], mode)
    return [res]
