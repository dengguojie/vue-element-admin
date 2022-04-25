#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for transformation
"""
# Third-Party Packages
import numpy
import tbetoolkits
from .registry import register_golden


@register_golden(["transpose_dsl"])
def _transpose_dsl(input0: numpy.ndarray, perm):
    return numpy.transpose(input0, perm)


@register_golden(["transpose", "transpose_d"])
def _transpose(context: "tbetoolkits.UniversalTestcaseStructure"):
    input_array0 = context.input_arrays[0]
    perm = context.other_runtime_params["perm"]
    return numpy.transpose(input_array0, perm)


@register_golden(["depth_to_space"])
def depth_to_space(context: "tbetoolkits.UniversalTestcaseStructure"):
    input_array0 = context.input_arrays[0]
    shapes = input_array0.shape
    block_size = context.other_compilation_params["block_size"]
    mode = context.other_compilation_params["mode"]
    data_format = context.other_compilation_params["data_format"]

    if mode == "DCR" and data_format == "NHWC":
        input_shapes = [shapes[0], shapes[1], shapes[2],
                        block_size, block_size, shapes[3] // block_size // block_size]
        perm = [0, 1, 3, 2, 4, 5]
    elif mode == "CRD" and data_format == "NHWC":
        input_shapes = [shapes[0], shapes[1], shapes[2],
                        shapes[3] // block_size // block_size, block_size, block_size]
        perm = [0, 1, 4, 2, 5, 3]
    elif mode == "DCR" and data_format == "NCHW":
        input_shapes = [shapes[0], block_size, block_size,
                        shapes[1] // block_size // block_size,
                        shapes[2], shapes[3]]
        perm = [0, 3, 4, 1, 5, 2]
    elif mode == "CRD" and data_format == "NCHW":
        input_shapes = [shapes[0], shapes[1] // block_size // block_size,
                        block_size, block_size, shapes[2], shapes[3]]
        perm = [0, 1, 4, 2, 5, 3]
    re_input_array = input_array0.reshape(input_shapes)
    return numpy.transpose(re_input_array, perm)


@register_golden(["dsl_gather", "gather"])
def dsl_gather(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    A numpy implementation of Gather
    """
    params_data = context.input_arrays[0]
    params_shape_len = len(params_data.shape)

    indices_data = context.input_arrays[1]
    indices_shape_len = len(indices_data.shape)

    if "batch_dims" in context.other_runtime_params:
        batch_dims = context.other_runtime_params["batch_dims"]
    else:
        batch_dims = 0

    batch_dims = batch_dims if batch_dims >= 0 else batch_dims + indices_shape_len

    axis = batch_dims

    axis = axis if axis >= 0 else axis + params_shape_len

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    # indices 1
    params_shape = params_data.shape
    indices_shape = indices_data.shape

    params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
    indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.gather(params, indices, axis=axis, batch_dims=batch_dims, name=None)
        res = sess.run(gather_res, feed_dict={params: params_data, indices: indices_data})
        return res


@register_golden(["dsl_gather_v2", "gather_v2"])
def dsl_gather_v2(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    A numpy implementation of Gather
    """
    params_data = context.input_arrays[0]
    params_shape_len = len(params_data.shape)

    indices_data = context.input_arrays[1]
    indices_shape_len = len(indices_data.shape)

    if "batch_dims" in context.other_runtime_params:
        batch_dims = context.other_runtime_params["batch_dims"]
    else:
        batch_dims = 0

    batch_dims = batch_dims if batch_dims >= 0 else batch_dims + indices_shape_len

    if "axis_dict" in context.other_runtime_params:
        axis = context.other_runtime_params["axis_dict"]
    else:
        if "axis" in context.other_runtime_params:
            axis = context.other_runtime_params["axis"]
        else:
            axis = 0

    axis = axis if axis >= 0 else axis + params_shape_len

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    # indices 1
    params_shape = params_data.shape
    indices_shape = indices_data.shape

    params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
    indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.gather(params, indices, axis=axis, batch_dims=batch_dims, name=None)
        res = sess.run(gather_res, feed_dict={params: params_data, indices: indices_data})

    return res


@register_golden(["dsl_gather_nd", "gather_nd"])
def dsl_gather(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    A numpy implementation of Gather
    """
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    params_data = context.input_arrays[0]
    indices_data = context.input_arrays[1]

    # indices 1
    params_shape = params_data.shape
    indices_shape = indices_data.shape

    params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
    indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.gather_nd(params, indices, name=None, batch_dims=0)

        res = sess.run(gather_res, feed_dict={params: params_data, indices: indices_data})

    return res


@register_golden(["pack"])
def concat(context: "tbetoolkits.UniversalTestcaseStructure"):
    axis = context.other_compilation_params["axis"]
    input_arrays = []
    if axis == -1 or axis == len(context.input_arrays[0].shape):
        for input_i in context.input_arrays:
            shape = list(input_i.shape)
            shape.append(1)
            input_arrays.append(input_i.reshape(shape))
    else:
        input_arrays = context.input_arrays
    return numpy.concatenate(input_arrays,
                             axis=context.other_compilation_params["axis"])


@register_golden(["concat_d", "concat_v2_d"])
def concat_d(context: "tbetoolkits.UniversalTestcaseStructure"):
    return numpy.concatenate(context.input_arrays,
                             axis=context.other_compilation_params["concat_dim"])


@register_golden(["dsl_slice", "slice"])
def dsl_slice(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x_data = context.input_arrays[0]
    begin_data = context.other_runtime_params["begin"]
    size_data = context.other_runtime_params["size"]

    # format info
    input_format = context.dyn_input_formats[0]
    ori_format = context.stc_input_ori_formats[0]
    org_shape = context.stc_ori_inputs[0]
    if input_format in ("NDC1HWC0", "NC1HWC0", "FRACTAL_NZ", "FRACTAL_Z", "FRACTAL_Z_3D"):
        begin_data, size_data = _update_params_for_other_format(org_shape, begin_data, size_data, input_format, ori_format)

    # indices 1
    x_shape = x_data.shape

    x = tf.compat.v1.placeholder(dtype=x_data.dtype, shape=x_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.slice(x, begin_data, size_data, name=None)

        res = sess.run(gather_res, feed_dict={x: x_data,})

    return res

def _handle_inputs_with_format(x, begin, size):

    x = update_shape_base_other_format(x)
    input_format = x.get("format")
    ori_format = x.get("ori_format")
    ori_shape = x.get("ori_shape")
    begin_list = begin.get("const_value")
    size_list = size.get("const_value")
    new_begin, new_size = _update_params_for_other_format(ori_shape, begin_list, size_list, input_format, ori_format)

    return new_begin, new_size


def _update_params_for_other_format(shape, begin, size, input_format, ori_format):
    """
    update begin, size base on  ori_format
    """
    # modify size base size value if value = -1 size = shape - begin
    size_new = []
    for i, item in enumerate(size):
        if item != -1:
            size_new.append(item)
        else:
            size_new.append(shape[i] - begin[i])
    size = size_new
    align_c0 = 16
    begin = list(begin)
    size = list(size)
    if input_format in ["NDC1HWC0", "NC1HWC0", "FRACTAL_Z", "FRACTAL_Z_3D"]:
        # when NDC1HWC0 or NC1HWC0 will update the C1 and C0 for begin and size
        # ex: begin [N, D, C, H, W] -> [N, D, C // 16, H, W, 0]
        #     size  [N, D, C, H, W] -> [N, D, (C + 15) // 16, H, W, -1]
        # when FRACTAL_Z or FRACTAL_Z_3D will update the C1 and C0 and N1 and N0
        # ex: begin [N, D, C, H, W] -> [D, C // 16, H, W, N // 16, 0, 0]
        #     size  [N, D, C, H, W] -> [D, (C + 15) // 16, H, W, (N + 15) // 16, 0, 0]
        begin_nchw = [begin[ori_format.index("N")], begin[ori_format.index("C")],
                      begin[ori_format.index("H")], begin[ori_format.index("W")]]
        size_nchw = [size[ori_format.index("N")], size[ori_format.index("C")],
                     size[ori_format.index("H")], size[ori_format.index("W")]]
        begin_c1 = begin_nchw[1] // align_c0
        begin_c0 = 0
        begin_n1 = begin_nchw[0] // align_c0
        begin_n0 = 0
        size_c1 = _ceil_div(size_nchw[1], align_c0)
        size_c0 = -1
        size_n1 = _ceil_div(size_nchw[0], align_c0)
        size_n0 = -1

        if input_format == "NDC1HWC0":
            begin_new = [begin_nchw[0], begin[ori_format.index("D")],
                         begin_c1, begin_nchw[2], begin_nchw[3], begin_c0]
            size_new = [size_nchw[0], size[ori_format.index("D")],
                        size_c1, size_nchw[2], size_nchw[3], size_c0]
        elif input_format == "NC1HWC0":
            begin_new = [begin_nchw[0], begin_c1, begin_nchw[2], begin_nchw[3], begin_c0]
            size_new = [size_nchw[0], size_c1, size_nchw[2], size_nchw[3], size_c0]
        elif input_format == "FRACTAL_Z_3D":
            begin_new = [begin[ori_format.index("D")],
                         begin_c1, begin_nchw[2], begin_nchw[3], begin_n1, begin_n0, begin_c0]
            size_new = [size[ori_format.index("D")],
                        size_c1, size_nchw[2], size_nchw[3], size_n1, size_n0, size_c0]
        else:
            begin_new = [begin_c1, begin_nchw[2], begin_nchw[3], begin_n1, begin_n0, begin_c0]
            size_new = [size_c1, size_nchw[2], size_nchw[3], size_n1, size_n0, size_c0]

        return begin_new, size_new

    if input_format in ["FRACTAL_NZ"]:
        # when FRACTAL_NZ will update last two dim
        # ex: begin [A, B, C, D] -> [A, B, D // 16,  C // 16, 0 , 0]
        #     size  [A, B, C, D] -> [A, B, (D + 15) // 16,  (C + 15) // 16, -1 , -1]
        begin_fisrt_last_dim_one = begin[-1] // align_c0
        begin_fisrt_last_dim_two = 0

        begin_second_last_dim_one = begin[-2] // align_c0
        begin_second_last_dim_two = 0

        size_fisrt_last_dim_one = _ceil_div(size[-1], align_c0)
        size_fisrt_last_dim_two = -1

        size_second_last_dim_one = _ceil_div(size[-2], align_c0)
        size_second_last_dim_two = -1

        begin_new = begin[0:-2] + [begin_fisrt_last_dim_one, begin_second_last_dim_one,
                                   begin_second_last_dim_two, begin_fisrt_last_dim_two]
        size_new = size[0:-2] + [size_fisrt_last_dim_one, size_second_last_dim_one,
                                 size_second_last_dim_two, size_fisrt_last_dim_two]

        return begin_new, size_new


def update_shape_base_other_format(input_dict):
    """
    update_axis_for_other_format: when format is changed, the axis will be updated
    """
    ori_shape = input_dict.get("ori_shape")
    ori_format = input_dict.get("ori_format")
    input_shape = input_dict.get("shape")
    input_format = input_dict.get("format")

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        # when FRACTAL_Z, mean: C1HWNiNoC0
        # when FRACTAL_Z_3D, mean: DC1HWNiNoC0
        if len(input_shape) == 4:
            # fe will reshape the C1HWNiNoC0/DC1HWNiNoC0 to 4s = (C1HW)NiNoC0/(DC1HW)NiNoC0
            # now will reshape to 6d/7d = C1HWNiNoC0/DC1HWNiNoC0
            dict_zip_shape = dict(zip(list(ori_format), ori_shape))
            shape_h_dim = dict_zip_shape["H"]
            shape_w_dim = dict_zip_shape["W"]

            shape_c1_dim = input_shape[0] // (shape_h_dim * shape_w_dim)
            new_shape = [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])
            if input_format == "FRACTAL_Z_3D":
                shape_d_dim = dict_zip_shape["D"]
                shape_c1_dim = new_shape[0] // shape_d_dim
                new_shape = [shape_d_dim] + [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])

            input_dict["shape"] = new_shape

    return input_dict


def _ceil_div(value, block):
    """
    integrate the input value by block

    """
    return (value + block - 1) // block
