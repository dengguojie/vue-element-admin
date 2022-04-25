#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Reduce series shape inference function"""
# Third-party Packages
import tbetoolkits
from .shape_func_registry import register_func
from ...utilities import eliminate_scalar_shapes


@register_func(["reduce_sum", "reduce_sum_d"])
def _reduce_sum(context: "tbetoolkits.UniversalTestcaseStructure"):
    if "axis" in context.other_compilation_params:
        axis = context.other_compilation_params["axis"]
    elif "axis" in context.other_runtime_params:
        axis = context.other_runtime_params["axis"]
    else:
        raise RuntimeError("Missing reduce axis")
    input_shape = list(eliminate_scalar_shapes(context.stc_inputs[0]))
    for _axis in axis:
        input_shape[_axis] = 1
    return (input_shape,)