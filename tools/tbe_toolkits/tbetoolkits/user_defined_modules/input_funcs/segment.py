#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Setment Input Generator"""
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


@register_input(["unsorted_segment_sum"])
def _unsorted_segment_sum_input(context: "tbetoolkits.UniversalTestcaseStructure") \
        -> Tuple[Tuple[Optional[numpy.ndarray], ...],
                 Tuple[Optional[numpy.ndarray], ...]]:
    segments = numpy.random.randint(low=0,
                                    high=context.other_compilation_params["num_segments"],
                                    size=context.input_arrays[1].size,
                                    dtype=context.input_arrays[1].dtype)
    segment_num = numpy.array([context.other_runtime_params["num_segments_dict"]],
                              dtype=tbetoolkits.utilities.get(context.dyn_input_dtypes, 2))
    return (context.input_arrays[0], segments, segment_num), (context.input_arrays[0], segments)
