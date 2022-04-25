#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""ReduceSum Input Generator"""
# Standard Packages
from typing import Tuple
from typing import Optional

# Third-Party Packages
import numpy
import tbetoolkits
from .registry import register_input


def __eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis]))
    return axis


@register_input(["reduce_sum",
                 "reduce_max",
                 "reduce_min",
                 "reduce_mean",
                 "reduce_prod",
                 "reduce_any",
                 "reduce_all"])
def _reduce_input(context: "tbetoolkits.UniversalTestcaseStructure") \
        -> Tuple[Tuple[Optional[numpy.ndarray], ...],
                 Tuple[Optional[numpy.ndarray], ...]]:
    ipt_0 = context.input_arrays[0]
    if "axis" in context.other_runtime_params:
        axes = context.other_runtime_params.get("axis")
    elif "axes" in context.other_runtime_params:
        axes = context.other_runtime_params.get("axes")
    else:
        raise RuntimeError("Reduce type operator missing axis parameter")
    if axes is None or len(axes) == 0:
        axes = tuple(range(len(ipt_0.shape)))
    else:
        axes = __eliminate_duplicate_axes(axes, ipt_0)
    ipt_1 = numpy.array(axes, dtype="int32")
    return (ipt_0, ipt_1), (ipt_0, None)
