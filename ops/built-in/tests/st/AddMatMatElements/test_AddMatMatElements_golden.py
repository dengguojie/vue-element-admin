import tensorflow as tf


def calc_expect_func(c, a, b, beta, alpha, c_out):
    c = c["value"]
    a = a["value"]
    b = b["value"]
    beta = beta["value"][0]
    alpha = alpha["value"][0]
    c_holder = tf.placeholder(c.dtype, shape=c.shape)
    a_holder = tf.placeholder(a.dtype, shape=a.shape)
    b_holder = tf.placeholder(b.dtype, shape=b.shape)
    ab = tf.multiply(a_holder, b_holder)
    ab_val = tf.multiply(ab, alpha)
    c_out = tf.multiply(c_holder, beta)
    res = tf.add(c_out, ab_val)
    with tf.Session() as sess:
        re = sess.run(res, feed_dict={c_holder: c, a_holder: a, b_holder: b})
    return [re]