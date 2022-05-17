import tensorflow as tf


def selugrad_by_tf(gradients, outputs):
    gradients_holder = tf.placeholder(gradients.dtype, shape=gradients.shape)
    outputs_holder = tf.placeholder(outputs.dtype, shape=outputs.shape)
    re = tf.raw_ops.SeluGrad(gradients=gradients_holder, outputs=outputs_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={gradients_holder: gradients, outputs_holder: outputs})
    return result

def calc_expect_func(gradients, outputs, y):
    res = selugrad_by_tf(gradients["value"], outputs["value"])
    return [res]