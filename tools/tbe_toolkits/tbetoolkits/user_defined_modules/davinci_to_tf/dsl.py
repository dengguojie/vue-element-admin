# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""pooling series tensorflow operators like max_pool"""
# Third-party Packages
import tbetoolkits
from . import dav2tf_registry


@dav2tf_registry.register_func(["dsl_concat_v2_d", "dsl_pack"])
def tensor_to_scalar(context: "tbetoolkits.UniversalTestcaseStructure"):
    context.op_name = context.op_name.replace("dsl_", "")
    return context

@dav2tf_registry.register_func(["gather_v2"])
def tensor_to_scalar(context: "tbetoolkits.UniversalTestcaseStructure"):
    axis_value = context.other_runtime_params["axis_dict"]
    context.other_runtime_params["axis"] = axis_value
    return context