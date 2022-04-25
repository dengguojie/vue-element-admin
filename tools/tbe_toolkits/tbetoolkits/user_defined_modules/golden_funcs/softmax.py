#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Golden function for softmax operator.
"""
# Standard Packages
import copy
from typing import Tuple
# Third-Party Packages
import numpy
from .registry import register_golden
# Local
from ...core_modules.infershape import format_transformation


def logsumexp(a, axis: Tuple = None, keepdims=False, return_sign=False):
    a_max = numpy.amax(a, axis=axis, keepdims=True)
    if a_max.ndim > 0:
        a_max[~numpy.isfinite(a_max)] = 0
    elif not numpy.isfinite(a_max):
        a_max = 0
    tmp = numpy.exp(a - a_max)

    with numpy.errstate(divide='ignore'):
        s = numpy.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = numpy.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = numpy.log(s)

    if not keepdims:
        a_max = numpy.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis=None):
    return numpy.exp(x - logsumexp(x, axis=axis, keepdims=True))


def normalize_axis(axis, shape_length) -> Tuple:
    normalized_axis = []
    if isinstance(axis, int):
        normalized_axis = [axis]
    elif isinstance(axis, tuple):
        normalized_axis = list(axis)
    elif isinstance(axis, list):
        normalized_axis = copy.deepcopy(axis)
    if not normalized_axis:
        normalized_axis = [-1]
    normalized_axis = [v if v >= 0 else v + shape_length for v in normalized_axis]
    normalized_axis = tuple(list(set(normalized_axis)))
    return normalized_axis


@register_golden(["softmax_v2"])
def softmax_golden(context):
    format = context.stc_input_formats[0]
    shape = context.stc_inputs[0]
    ori_format = context.stc_input_ori_formats[0]
    ori_shape = context.stc_ori_inputs[0]
    data = context.input_arrays[0]
    axis = context.other_compilation_params.get("axis")

    # the axis is corresponding to the original shape
    # normalize axis
    axis = normalize_axis(axis, len(ori_shape))

    # convert any format to ND
    if format == "NC1HWC0":
        data = format_transformation.fhd2nd(data, ori_shape, ori_format)
    elif format == "FRACTAL_NZ":
        data = format_transformation.nz2nd(data, ori_shape)

    # calc softmax
    result = softmax(data, axis)

    # convert ND to target format
    if format == "NC1HWC0":
        result = format_transformation.nd2fhd(result, ori_format)
    elif format == "FRACTAL_NZ":
        result = format_transformation.nd2nz(result)

    return result
