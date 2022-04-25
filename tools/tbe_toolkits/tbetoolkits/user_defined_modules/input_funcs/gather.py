#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Scatter input tensor generator"""
# Standard Packages
from typing import Tuple
from typing import Optional

# Third-Party Packages
import tbetoolkits
import numpy
from .registry import register_input


@register_input(["gather"])
def _gather_input(context: tbetoolkits.UniversalTestcaseStructure) \
        -> Tuple[Tuple[Optional[numpy.ndarray], ...],
                 Tuple[Optional[numpy.ndarray], ...]]:
    numpy.random.seed(0)
    if not context.input_arrays[0].dtype == "bool":
        ipt_0 = numpy.arange(0,
                             context.input_arrays[0].size,
                             1, dtype=context.input_arrays[0].dtype).reshape(context.input_arrays[0].shape)
    else:
        ipt_0 = numpy.random.choice(a=[False, True], size=context.input_arrays[0].shape,
                                    p=[0.5, 0.5]).reshape(context.input_arrays[0].shape)
    ipt_1 = numpy.random.default_rng().choice(ipt_0.shape[context.other_runtime_params.setdefault("axis", 0)],
                                              size=context.input_arrays[1].size)
    ipt_1 = ipt_1.astype(context.input_arrays[1].dtype)
    return (ipt_0, ipt_1), (ipt_0, ipt_1)


@register_input(["gather_v2"])
def _gather_v2_input(context: tbetoolkits.UniversalTestcaseStructure) \
        -> Tuple[Tuple[Optional[numpy.ndarray], ...],
                 Tuple[Optional[numpy.ndarray], ...]]:
    axis = context.other_runtime_params.setdefault("axis", 0)
    numpy.random.seed(0)
    if not context.input_arrays[0].dtype == "bool":
        ipt_0 = numpy.arange(0,
                             context.input_arrays[0].size,
                             1, dtype=context.input_arrays[0].dtype).reshape(context.input_arrays[0].shape)
    else:
        ipt_0 = numpy.random.choice(a=[False, True], size=context.input_arrays[0].shape,
                                    p=[0.5, 0.5]).reshape(context.input_arrays[0].shape)
    ipt_1 = numpy.random.default_rng(0).choice(ipt_0.shape[axis],
                                              size=context.input_arrays[1].size)
    ipt_1 = ipt_1.astype(context.input_arrays[1].dtype)
    ipt_1 = numpy.reshape(ipt_1.astype(context.input_arrays[1].dtype), context.input_arrays[1].shape)

    if len(context.dyn_input_dtypes) >= 3:
        ipt_2 = numpy.array([axis], dtype=context.dyn_input_dtypes[2])
    else:
        ipt_2 = numpy.array([0], dtype="int32")
    return (ipt_0, ipt_1, ipt_2), (ipt_0, ipt_1)
