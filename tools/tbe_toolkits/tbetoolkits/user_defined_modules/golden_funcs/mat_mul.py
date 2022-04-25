# Standard Packages
# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Mat_mul Golden"""
# Standard Packages
import math

# Third-Party Packages
import tbetoolkits
import numpy as np
from .registry import register_golden
from ...utilities import get


def fractal_shape(dtype):
    """
    >>> fractal_shape('int8')
    (16, 32)
    >>> fractal_shape('float16')
    (16, 16)
    >>> fractal_shape('float32')
    (16, 8)
    """
    import re
    res = re.match("[^\d]+(\d+)", dtype)
    assert res is not None
    bit_of_dtype = int(res[1])
    assert (32 * 8) % bit_of_dtype == 0

    return 16, (32 * 8) // bit_of_dtype


def shape_nd_to_Nz(shape, dtype='float16', before_mmad=True):
    """
    >>> shape_nd_to_Nz([3,17])
    [2, 1, 16, 16]
    >>> shape_nd_to_Nz([4,5,3,17])
    [4, 5, 2, 1, 16, 16]
    >>> shape_nd_to_Nz([3,17], dtype='int8')
    [1, 1, 16, 32]
    >>> shape_nd_to_Nz([16,27], dtype='int32')
    [4, 1, 16, 8]
    >>> shape_nd_to_Nz([16,27], dtype='int32', before_mmad=False)
    [2, 1, 16, 16]
    """
    assert (dtype, before_mmad) in (
        ('float16', True), ('float32', False), ('int8', True),
        ('int32', False), ('int32', True),('float64', False)
    ), f"Please implement shape ND to FRACTAL_NZ with dtype {dtype} on {shape} {'before mmad' if before_mmad else 'after mmad'}"

    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]

    if before_mmad:
        a0, b0 = fractal_shape(dtype)
    else:
        a0, b0 = 16, 16

    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def transpose_input(x, trans):
    if trans:
        array_trans = gen_axes_for_transpose(len(x.shape) - 2, [1, 0])
        return x.transpose(*array_trans)
    return x

def due_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


@register_golden(["batch_matmul_v2", "batch_matmul", "mat_mul"])
def _matmul(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x1, x2, bias, *_ = context.input_arrays
    trans_a = context.other_runtime_params.get("trans_a")
    trans_b = context.other_runtime_params.get("trans_b")
    format_x1 = get(context.stc_input_formats, 0)
    format_x2 = get(context.stc_input_formats, 1)
    format_out = context.output_formats[0]
    output_dtype = context.output_dtypes
    if format_x1 == 'FRACTAL_NZ':
        # b, m, k -Nz-> b, k1, m1, m0, k0
        array_trans = gen_axes_for_transpose(len(x1.shape) - 4, [1, 2, 0, 3])
        *_, k1, m1, m0, k0 = x1.shape
        batch_x1 = x1.shape[:-4]
        array_reshape = list(batch_x1) + [m1 * m0, k1 * k0]
        x1 = x1.transpose(*array_trans).reshape(*array_reshape)

    if format_x2 == 'FRACTAL_NZ':
        # b, k, n -Nz-> b, n1, k1, k0, n0
        array_trans = gen_axes_for_transpose(len(x2.shape) - 4, [1, 2, 0, 3])
        *_, n1, k1, k0, n0 = x2.shape
        batch_x2 = x2.shape[:-4]
        array_reshape = list(batch_x2) + [k1 * k0, n1 * n0]
        x2 = x2.transpose(array_trans).reshape(*array_reshape)

    if format_x2 == 'FRACTAL_ZN_RNN':
        # b, k, n -Z -> b, k1, n1, n0, k0
        array_trans = gen_axes_for_transpose(len(x2.shape) - 4, [0, 3, 1, 2])
        *_, k1, n1, n0, k0 = x2.shape
        batch_x2 = x2.shape[:-4]
        array_reshape = list(batch_x2) + [k1 * k0, n1 * n0]
        x2 = x2.transpose(array_trans).reshape(*array_reshape)


    a_data = x1.astype('float32')
    b_data = x2.astype('float32')
    a = tf.compat.v1.placeholder(a_data.dtype, shape=a_data.shape)
    b = tf.compat.v1.placeholder(b_data.dtype, shape=b_data.shape)
    res_tf = tf.compat.v1.matmul(a, b, transpose_a=trans_a, transpose_b=trans_b)
    bias_data = None
    if bias is not None:
        matmul_result_shape = res_tf.shape
        n_out = matmul_result_shape[-1]
        bias_data = bias.astype('float32')
        bias_ori_shape = bias.shape
        bias_data = np.pad(bias_data, ((0, int(n_out - bias_ori_shape[0]))), 'constant', constant_values=(0,0))
        bias = tf.compat.v1.placeholder(bias_data.dtype, shape=n_out)
        res_tf = tf.compat.v1.add(res_tf, bias)

    with tf.compat.v1.Session() as sess:
        feed_dict = {
            a: a_data,
            b: b_data,
        }
        if bias is not None:
            feed_dict[bias] = bias_data
        res_tf = sess.run(res_tf, feed_dict=feed_dict)
    if format_out == 'ND':
        if output_dtype[0] == 'float16':
            res_tf = due_overflow(res_tf.astype(np.float16))
    elif format_out == 'FRACTAL_NZ':
        array_trans = gen_axes_for_transpose(len(res_tf.shape) - 2, [2, 0, 1, 3])
        y_shape = shape_nd_to_Nz(res_tf.shape,
                                 dtype=res_tf.dtype,
                                 before_mmad=False)
        *_, n1, m1, m0, n0 = y_shape
        res_tf = res_tf.reshape(y_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)
        if output_dtype[0] == 'float16':
            res_tf = due_overflow(res_tf.astype(np.float16))

    return res_tf

@register_golden(["gemm"])
def _gemm(context: "tbetoolkits.UniversalTestcaseStructure"):
    x1, x2, bias, alpha, beta = context.input_arrays
    trans_a = context.other_runtime_params.get("trans_a")
    trans_b = context.other_runtime_params.get("trans_b")
    format_x1, format_x2, format_bias, format_alpha, format_beta = context.stc_input_formats
    format_out = context.output_formats[0]
    stc_input_dtypes = context.stc_input_dtypes
    output_dtype = context.output_dtypes[0]
    if format_x1 == 'FRACTAL_NZ':
        # k1, m1, m0, k0 ->m, k
        array_trans = gen_axes_for_transpose(len(x1.shape) - 4, [1, 2, 0, 3])
        *_, k1, m1, m0, k0 = x1.shape
        batch_x1 = x1.shape[:-4]
        array_reshape = list(batch_x1) + [m1 * m0, k1 * k0]
        x1 = x1.transpose(*array_trans).reshape(*array_reshape)

    if format_x2 == 'FRACTAL_NZ':
        # n1, k1, k0, n0-> k, n
        array_trans = gen_axes_for_transpose(len(x2.shape) - 4, [1, 2, 0, 3])
        *_, n1, k1, k0, n0 = x2.shape
        batch_x2 = x2.shape[:-4]
        array_reshape = list(batch_x2) + [k1 * k0, n1 * n0]
        x2 = x2.transpose(*array_trans).reshape(*array_reshape)

    if format_x2 == 'FRACTAL_Z':
        # k1, n1, n0, k0-> k, n
        array_trans = gen_axes_for_transpose(len(x2.shape) - 4, [0, 1, 2, 3])
        *_, k1, n1, n0, k0 = x2.shape
        batch_x2 = x2.shape[:-4]
        array_reshape = list(batch_x2) + [k1 * k0, n1 * n0]
        x2 = x2.transpose(*array_trans).reshape(*array_reshape)

    a = transpose_input(x1, trans_a)
    b = transpose_input(x2, trans_b)
    a = a.astype('float32')
    b = b.astype('float32')
    res_nd = alpha*np.matmul(a, b)
    m_out, n_out = res_nd.shape[-2:]
    if format_bias == 'FRACTAL_NZ':
        # n1, m1, m0, n0-> m, n
        array_trans = gen_axes_for_transpose(len(bias.shape) - 4, [1, 2, 0, 3])
        *_, n1, m1, m0, n0 = bias.shape
        batch_bias = x2.shape[:-4]
        array_reshape = list(batch_bias) + [m1 * m0, n1 * n0]
        bias = bias.transpose(*array_trans).reshape(*array_reshape)
    else:
        if stc_input_dtypes[2] == 'int32':
            m_ori, n_ori = bias.shape
            bias = np.pad(bias, ((0, m_out - m_ori), (0, n_out - n_ori)), 'constant')

    bias = bias.astype('float32')
    res_nd += beta*bias
    if format_out == 'ND':
        if output_dtype == 'float16':
            res_nd = due_overflow(res_nd.astype(np.float16))
    elif format_out == 'FRACTAL_NZ':
        y_shape = shape_nd_to_Nz(res_nd.shape,
                                 dtype=res_nd.dtype,
                                 before_mmad=False)
        n1, m1, m0, n0 = y_shape
        res_nd = res_nd.reshape([m1, m0, n1, n0]).transpose([2, 0, 1, 3])
        if output_dtype == 'float16':
            res_nd = due_overflow(res_nd.astype(np.float16))
    return res_nd

