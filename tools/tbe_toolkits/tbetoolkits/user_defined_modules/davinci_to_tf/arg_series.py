# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Arg series tensorflow operators like arg_max_v2"""
# Third-party Packages
import tbetoolkits
from .dav2tf_registry import register_func


@register_func(["arg_max_v2"])
def _arg_max_v2(context: "tbetoolkits.UniversalTestcaseStructure"):
    if "dimension" in context.other_compilation_params:
        params = context.other_compilation_params
    elif "dimension" in context.other_runtime_params:
        params = context.other_runtime_params
    else:
        return context
    axis = params["dimension"]
    del params["dimension"]
    params["axis"] = axis
    return context
