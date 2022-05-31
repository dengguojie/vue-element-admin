import tensorflow as tf


def asin_grad_compute(y, dy, z, kernel_name="asin_grad"):
    y = y.get("value")
    dy = dy.get("value")
    num_minus_one = -1
    num_one = 1
    dtype = y.dtype
    # `step 1: calculate num_to_vrsqrt = 1 - y^2`
    data = tf.multiply(y, y)
    data = data * num_minus_one
    num_to_vrsqrt = data + num_one

    # step 2: calculate dy * (1 / sqrt(1 - y^2))
    vsqrt_res = tf.sqrt(num_to_vrsqrt)
    res = tf.div(dy, vsqrt_res)

    with tf.Session() as sess:
        res = sess.run(res)
    return res