import numpy as np
import tensorflow as tf


def resize_by_tf(x, size, align_corners, half_pixel_centers):
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.compat.v1.image.resize_bilinear(x_holder, size,
                                            align_corners=align_corners,
                                            half_pixel_centers=half_pixel_centers)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result


def calc_expect_func(x, size, y, align_corners, half_pixel_centers):
    res = resize_by_tf(x["value"], size["value"], align_corners, half_pixel_centers)
    return [res]
