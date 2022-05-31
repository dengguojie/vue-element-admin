import tensorflow as tf


def asinh_compute(input_x, output_y, kernel_name="asinh"):
    output = tf.math.asinh(input_x.get("value"))
    with tf.Session() as sess:
        res = sess.run(output)
    return [res]