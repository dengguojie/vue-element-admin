import tensorflow as tf


def calc_expect_func(y, dy, z):
    y = y["value"]
    dy = dy["value"]
    y_holder = tf.placeholder(y.dtype, shape=y.shape)
    dy_holder = tf.placeholder(dy.dtype, shape=dy.shape)
    x1 = tf.sinh(y_holder)
    x2 = 1 / x1
    re = tf.multiply(dy_holder, x2)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={y_holder: y, dy_holder:dy})
    return [result]