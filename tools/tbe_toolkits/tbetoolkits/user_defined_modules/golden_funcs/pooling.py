#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for pooling
"""
# Third-Party Packages
import tbetoolkits
from .registry import register_golden
from tensorflow.python.ops import gen_nn_ops
import numpy as np


@register_golden(["max_pool_grad"])
def maxpoolgrad(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    ksizes = context.other_compilation_params.get("ksize")
    strides = context.other_compilation_params.get("strides")
    padding = context.other_compilation_params.get("padding")
    input_shape, input_orig_y_shape, grad_shape = context.stc_ori_inputs[:3]
    input = tf.placeholder(shape=input_shape, dtype='float16')
    input_orig_y = tf.placeholder(shape=input_orig_y_shape, dtype='float16')
    grad = tf.placeholder(shape=grad_shape, dtype='float16')
    # 由于TBE算子的输入是nc1hwc0格式，对标的tf算子的输入是nhwc格式，框架的输入是按TBE的规格生成，所以需做转换
    input_data = nc1hwc0_2_nhwc(context.input_arrays[0], input_shape)
    input_orig_y_data = nc1hwc0_2_nhwc(context.input_arrays[1], input_orig_y_shape)
    grad_data = nc1hwc0_2_nhwc(context.input_arrays[2], grad_shape)

    maxpoolgrad_res = gen_nn_ops.max_pool_grad(input, input_orig_y, grad, ksizes, strides,
                                               padding, data_format="NHWC", name="maxpoolgrad")
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(maxpoolgrad_res, feed_dict={input: input_data, input_orig_y: input_orig_y_data, grad: grad_data})
        # 由于TBE算子的输出是nc1hwc0格式，对标的tf算子的输出是nhwc格式，所以标杆算完后需做转换
    res = nhwc_2_nc1hwc0(res, 'float16')
    return res.astype(context.output_dtypes[0])


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
