#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""fill input tensor generator"""
# Standard Packages

# Third-Party Packages
import numpy
import tbetoolkits
from .registry import register_input


@register_input(["fill"])
def _fill_input(context: tbetoolkits.UniversalTestcaseStructure):
    ipt_0 = numpy.array(context.other_runtime_params["dims"], dtype="int32")
    ipt_1 = context.input_arrays[1 - 1]
    return (ipt_0, ipt_1), (None, ipt_1)
