import tensorflow as tf


def asin_compute(x, y, kernel_name="asin"):
    output = tf.math.asin(x.get("value"))
    with tf.Session() as sess:
        res = sess.run(output)
    return [res]