#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for elementary reduce pattern
"""
# Third-Party Packages
import numpy
import tbetoolkits
from .registry import register_golden


def __eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis]))
    return axis


@register_golden(["bn_training_update_grad", "bn_training_update_grad_dev"])
def bn_training_update_grad(context: "tbetoolkits.UniversalTestcaseStructure"):
    axis = (0, 2, 3)
    grads, x, batch_mean, batch_variance = context.input_arrays
    grads = grads.astype(numpy.float64)
    x = x.astype(numpy.float64)
    batch_mean = batch_mean.astype(numpy.float64)
    batch_variance = batch_variance.astype(numpy.float64)

    epsilon = context.other_compilation_params.get("epsilon")
    x_norm = (x - batch_mean) / numpy.sqrt(batch_variance + epsilon)
    diff_scale, diff_offset = numpy.sum(x_norm * grads, axis=axis), numpy.sum(grads, axis=axis)
    return diff_scale.astype("float32"), diff_offset.astype("float32")


@register_golden(["bn_training_reduce", "bn_training_reduce_dev"])
def bn_training_reduce(context: "tbetoolkits.UniversalTestcaseStructure"):
    axis = (0, 2, 3)
    x = context.input_arrays[0]
    x = x.astype(numpy.float64)
    square_x = x * x
    _sum = numpy.sum(x, axis=axis)
    _square_sum = numpy.sum(square_x, axis=axis)
    return _sum.astype("float32"), _square_sum.astype("float32")


@register_golden(["reduce_sum", "reduce_sum_d"])
def _reduce_sum_d(context: "tbetoolkits.UniversalTestcaseStructure"):
    axis = context.other_runtime_params.get("axes")
    if axis is None:
        axis = context.other_runtime_params.get("axis")
    axis = __eliminate_duplicate_axes(axis, context.input_arrays[0])
    return numpy.sum(context.input_arrays[0], axis=axis)


@register_golden(["reduce_max", "reduce_max_d"])
def _reduce_max_d(context):
    axis = context.other_runtime_params.get("axes")
    if axis is None:
        axis = context.other_runtime_params.get("axis")
    axis = __eliminate_duplicate_axes(axis, context.input_arrays[0])
    return numpy.max(context.input_arrays[0], axis=axis)


@register_golden(["reduce_min", "reduce_min_d"])
def _reduce_min_d(context):
    axis = context.other_runtime_params.get("axes")
    if axis is None:
        axis = context.other_runtime_params.get("axis")
    axis = __eliminate_duplicate_axes(axis, context.input_arrays[0])
    return numpy.min(context.input_arrays[0], axis=axis)


@register_golden(["reduce_prod", "reduce_prod_d"])
def _reduce_prod_d(context):
    axis = context.other_runtime_params.get("axes")
    if axis is None:
        axis = context.other_runtime_params.get("axis")
    axis = __eliminate_duplicate_axes(axis, context.input_arrays[0])
    return numpy.prod(context.input_arrays[0], axis=axis)


@register_golden(["reduce_mean", "reduce_mean_d"])
def _reduce_mean_d(context):
    axis = context.other_runtime_params.get("axes")
    if axis is None:
        axis = context.other_runtime_params.get("axis")
    axis = __eliminate_duplicate_axes(axis, context.input_arrays[0])
    return numpy.mean(context.input_arrays[0], axis=axis)


@register_golden(["reduce_any", "reduce_any_d"])
def _reduce_any_d(context):
    axis = context.other_runtime_params.get("axes")
    if axis is None:
        axis = context.other_runtime_params.get("axis")
    axis = __eliminate_duplicate_axes(axis, context.input_arrays[0])
    return numpy.any(context.input_arrays[0], axis=axis)


@register_golden(["reduce_all", "reduce_all_d"])
def _reduce_all_d(context):
    axis = context.other_runtime_params.get("axes")
    if axis is None:
        axis = context.other_runtime_params.get("axis")
    axis = __eliminate_duplicate_axes(axis, context.input_arrays[0])
    return numpy.all(context.input_arrays[0], axis=axis)


def _infer_axes(input_data_format, data_format, shape):
    """
    To infer sum operate axis by input_data format and data_format
    to keep compute Architecture, so use global parameter send variable
    Parameters:
    ----------
    input_data_format: str
        op's input data format
    data_format: str
        'NCHW' or 'NHWC'
    shape : tuple or list
        the input data shape

    Returns
    -------
    g_shape_list. list
    """
    g_shape_list = []
    if input_data_format == 'FRACTAL_NZ':
        if data_format == "NCHW":
            if len(shape) == 4:
                for i in range(-1 * len(shape), 0):
                    if i not in (-1, -4):
                        g_shape_list += [i + len(shape)]
            elif len(shape) == 5:
                for i in range(-1 * len(shape), 0):
                    if i not in (-2, -3):
                        g_shape_list += [i + len(shape)]
            else:
                g_shape_list.append(0)
                for i in range(2, len(shape)):
                    # noinspection PyAugmentAssignment
                    g_shape_list = g_shape_list + [i]
        else:
            if len(shape) < 4:
                raise RuntimeError("cce_bias_add_grad_nz_2_nhwc only support shape larger than 4D")
            for i in range(-1 * len(shape), 0):
                if i not in (-1, -4):
                    g_shape_list += [i + len(shape)]
    elif input_data_format in ("FRACTAL_Z", "FRACTAL_Z_3D", "NC1HWC0", "NDC1HWC0"):
        if input_data_format == "FRACTAL_Z":
            # mean format is FRACTAL_Z, shape is C1HWNiNoC0
            g_shape_list = [1, 2, 3, 4]
        elif input_data_format == "FRACTAL_Z_3D":
            # mean format is FRACTAL_Z_3D, shape is DC1HWNiNoC0
            g_shape_list = [0, 2, 3, 4, 5]
        elif input_data_format == "NC1HWC0":
            # mean format is NC1HWC0, shape is NC1HWC0
            g_shape_list = [0, 2, 3]
        elif input_data_format == "NDC1HWC0":
            # mean format is NDC1HWC0, shape is NDC1HWC0
            g_shape_list = [0, 1, 3, 4]
    else:
        if data_format == "NCHW":
            g_shape_list = [0]
            for i in range(2, len(shape)):
                g_shape_list += [i]
        else:
            if len(shape) < 2:
                raise RuntimeError("cce_bias_add_grad only support shape larger than 2D")
            g_shape_list = [x for x in range(len(shape) - 1)]

    return g_shape_list


@register_golden(["bias_add_grad"])
def bias_add_grad(input_tensor, actual_formats, data_format):
    """
    Golden for bias_add_grad
    :param input_tensor:
    :param actual_formats:
    :param data_format:
    :return:
    """
    axis = __eliminate_duplicate_axes(_infer_axes(actual_formats[0], data_format, input_tensor.shape), input_tensor)
    dtype = input_tensor.dtype
    if dtype == "float32":
        return numpy.sum(input_tensor, axis=axis, dtype="float64").astype(dtype)
    else:
        return numpy.sum(input_tensor, axis=axis)
