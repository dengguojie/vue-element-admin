import tensorflow as tf


def selugrad_by_tf(y_grad, y):
    gradients_holder = tf.placeholder(y_grad.dtype, shape=y_grad.shape)
    outputs_holder = tf.placeholder(y.dtype, shape=y.shape)
    re = tf.raw_ops.SeluGrad(gradients=gradients_holder, outputs=outputs_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={gradients_holder: y_grad, outputs_holder: y})
    return result

def calc_expect_func(y_grad, y, x_grad):
    res = selugrad_by_tf(y_grad["value"], y["value"])
    return [res]