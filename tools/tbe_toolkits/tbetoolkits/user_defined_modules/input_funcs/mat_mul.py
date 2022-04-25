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


@register_input(["mat_mul"])
def _matmul_input(context: "tbetoolkits.core_modules.dynamic_shape.ProfilingContextStructure"):
    print("============================================register_input_matmul=========================")
    is_gpu = get_global_storage().mode.is_gpu()
    if is_gpu:
        return (context.input_arrays[0],
                context.input_arrays[1],),(context.input_arrays[0],context.input_arrays[1],)
    else:
        return context.input_arrays, context.input_arrays
