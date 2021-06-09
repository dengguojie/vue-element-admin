#！/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for dx pattern
'''
#Third-Party Packages
import tensorflow as tf


def calc_expect_func(x, out_backprop, y, filter_size, strides, padds, dilations, groups, data_format="NCHW"):
    xshape = x.get("ori_shape")
    xnhwc_shape = (xshape[0], xshape[2], xshape[3], xshape[1])
    x_data = x.get("value")
    xnhwc_data = x_data.transpose(0, 2, 3, 1).astype("float32")
    xnhwc = tf.compat.v1.placeholder(tf.float16, xnhwc_shape)
    xnhwc = tf.cast(xnhwc, tf.float32)

    dyshape = out_backprop.get("ori_shape")
    dynhwc_shape = (dyshape[0], dyshape[2], dyshape[3], dyshape[1])
    dy_data = filter.get("value")
    dynhwc_data = dy_data.transpose(0, 2, 3, 1).astype("float32")
    dynhwc = tf.compat.v1.placeholder(tf.float16, dynhwc_shape)
    dynhwc = tf.cast(dynhwc, tf.float32)

    wshape = filter.get("ori_shape")
    whwcn_shape = (wshape[2], wshape[3], wshape[1], wshape[0])
    strides = (strides[0], strides[2], strides[3],strides[1])

    padding = "VALID"
    for i in padds:
        if i != 0:
            padding = "SAME"

    dw = tf.nn.conv2d_backprop_input(input=xnhwc,
        filter_sizes=whwcn_shape,
        out_backprop=dynhwc,
        strides = strides,
        data_format='NHWC',
        padding=padding,
    )
    res = tf.transpose(dw, (3,2,0,1))
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        result = sess.run(res, feed_dict={xnhwc: xnhwc_data, dynhwc: dynhwc_data})
    return [result]
