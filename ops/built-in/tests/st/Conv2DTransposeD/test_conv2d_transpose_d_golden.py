#！/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for dx pattern
'''
#Third-Party Packages
import tensorflow as tf


def calc_expect_func(out_backprop, filter, y, input_size, strides, padds, dilations, groups, data_format="NCHW"):
    wshape = filter.get("ori_shape")
    whwcn_shape = (wshape[2], wshape[3], wshape[1], wshape[0])
    w_data = filter.get("value")
    whwcn_data = w_data.transpose(2, 3, 1, 0)
    whwcn = tf.compat.v1.placeholder(tf.float16, whwcn_shape)
    whwcn = tf.cast(whwcn, tf.float32)

    dyshape = out_backprop.get("ori_shape")
    dynhwc_shape = (dyshape[0], dyshape[2], dyshape[3], dyshape[1])
    dy_data = filter.get("value")
    dynhwc_data = w_data.transpose(0, 2, 3, 1).astype("float32")
    dynhwc = tf.compat.v1.placeholder(tf.float16, dynhwc_shape)
    dynhwc = tf.cast(dynhwc, tf.float32)

    dxshape = y.get("ori_shape")
    dxnhwc_shape = (dxshape[0], dxshape[2], dxshape[3], dxshape[1])
    strides = (strides[0], strides[2], strides[3],strides[1])

    padding = "VALID"
    for i in padds:
        if i != 0:
            padding = "SAME"

    dx = tf.nn.conv2d_backprop_input(input_sizes=dxnhwc_shape,
        filter=whwcn,
        out_backprop=dynhwc,
        strides = strides,
        data_format='NHWC',
        padding=padding,
    )
    dx = tf.cast(dx, tf.float16)
    res = tf.transpose(dx,(0,3,1,2))
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        result = sess.run(res, feed_dict={whwcn: whwcn_data, dynhwc: dynhwc_data})
    return [result]
