import tensorflow as tf


def calc_expect_func(x, y):
    x = x["value"]
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.acosh(x=x_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return [result]