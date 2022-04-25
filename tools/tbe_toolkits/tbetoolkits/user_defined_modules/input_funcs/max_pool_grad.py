#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Conv2d input tensor generator"""
# Standard Packages
from typing import Tuple
from typing import Optional
from ...utilities import get_global_storage
# Third-Party Packages
import numpy
from .registry import register_input


@register_input(["max_pool_grad"])
def _maxpoolgrad_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    input_data, input_orig_y_data, grad_data = context.input_arrays[:3]
    ksizes = context.other_compilation_params.get("ksize")
    strides = context.other_compilation_params.get("strides")
    padding = context.other_compilation_params.get("padding")
    input_shape = context.stc_ori_inputs[0]
    dtype = context.stc_input_dtypes[0]

    input_data_4d = nc1hwc0_2_nhwc(input_data, input_shape)
    input = tf.placeholder(shape=input_shape, dtype=dtype)
    input_orig_y = tf.nn.max_pool(input, ksizes, strides, padding, name="maxpool")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        ori_y_output = sess.run(input_orig_y, feed_dict={input: input_data_4d})
    input_orig_y_data_5d = nhwc_2_nc1hwc0(ori_y_output, dtype)
    return (input_data, input_orig_y_data_5d, grad_data), (input_data, input_orig_y_data_5d, grad_data)


def nc1hwc0_2_nhwc(array, shape_to):
    import numpy as np
    shape_from = array.shape
    axis_n = shape_from[0]
    axis_c1 = shape_from[1]
    axis_h = shape_from[2]
    axis_w = shape_from[3]
    axis_c0 = shape_from[4]
    # 如果四维的C轴被16整除则不用管C轴多补的pad，否则要对多余的C轴位置截断
    c_pad = None if axis_c1 * axis_c0 == shape_to[3] else shape_to[3] - axis_c1 * axis_c0
    array_shape = array.reshape(axis_n, axis_c1, axis_h, axis_w, axis_c0)
    tmp_input_tensor = np.transpose(array_shape, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_n, axis_h, axis_w, axis_c1 * axis_c0))
    # 返回截断C轴多补的pad
    return tmp_input_tensor[:, :, :, :c_pad]


def nhwc_2_nc1hwc0(array, dtype):
    import numpy as np
    shape_from = array.shape
    axis_n = shape_from[0]
    axis_h = shape_from[1]
    axis_w = shape_from[2]
    axis_c = shape_from[3]
    c0 = 32 if dtype == 'int8' or dtype == 'uint8' else 16
    c1 = (axis_c + c0 - 1) // c0
    x_pad = np.zeros((axis_n, c1 * c0, axis_h, axis_w), dtype)
    tmp_array = array.reshape(shape_from)
    # 先把nhwc转成nchw
    tmp_array = np.transpose(tmp_array, axes=(0, 3, 1, 2))
    # 填充C轴前axis_c个数据，之后的还是0
    x_pad[:, :axis_c, :, :] = tmp_array
    # nchw reshape成NC1C0HW，再转成nc1hwc0
    tmp_array = x_pad.reshape(axis_n, c1, c0, axis_h, axis_w).transpose(0, 1, 3, 4, 2).copy()
    return tmp_array


def nc1hwc0_pad0_input(input_data, shape_orig):
    import numpy as np
    fmi = input_data
    fmi_shape = np.shape(input_data)
    for n in range(fmi_shape[0]):
        for c1 in range(fmi_shape[1]):
            for h in range(fmi_shape[2]):
                for w in range(fmi_shape[3]):
                    for c0 in range(fmi_shape[4]):

                        if c1 * fmi_shape[4] + c0 >= shape_orig[3]:
                            fmi[n][c1][h][w][c0] = 0
    return fmi
