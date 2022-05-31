import tensorflow as tf


def calc_expect_func(x1, x2, y):
    x1 = x1["value"]
    x2 = x2["value"]

    x1_holder = tf.placeholder(x1.dtype, shape=x1.shape)
    x2_holder = tf.placeholder(x2.dtype, shape=x2.shape)
    
    res = tf.bitwise.bitwise_xor(x1_holder, x2_holder)
    with tf.Session() as sess:
        re = sess.run(res, feed_dict={x1_holder: x1, x2_holder: x2})
    return [re]