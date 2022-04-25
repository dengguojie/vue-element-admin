#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Davinci operator parameter transformation map
"""
# Standard Packages
from typing import Sequence

# Third-party Packages
import tbetoolkits
from . import dav2tf_registry
from . import arg_series
from . import dsl
from . import pooling
from . import conv2d


@dav2tf_registry.register_func(["softmax_v2"])
def tensor_to_scalar(context: "tbetoolkits.UniversalTestcaseStructure"):
    """Davinci operators tend to use tensor as axis input, while tf ops don't"""
    if "axis" in context.other_compilation_params:
        params = context.other_compilation_params
    elif "axis" in context.other_runtime_params:
        params = context.other_runtime_params
    else:
        return context
    axis = params["axis"]
    if isinstance(axis, Sequence):
        axis = axis[0]
    params["axis"] = axis
    return context


@dav2tf_registry.register_func(["logical_or", "logical_not", "logical_and",
                                "reduce_all", "reduce_any",
                                "reduce_all_d", "reduce_any_d", "select"])
def logical(context: "tbetoolkits.UniversalTestcaseStructure"):
    """logical dtype is int8 for davinci"""
    context.stc_input_dtypes = ["bool" if dtype in ["int8", "uint8"] else dtype for dtype in context.stc_input_dtypes]
    if "keep_dims" in context.other_compilation_params:
        params = context.other_compilation_params
    elif "keep_dims" in context.other_runtime_params:
        params = context.other_runtime_params
    else:
        return context
    if "keep_dims" in params:
        params["keepdims"] = params["keep_dims"]
        del params["keep_dims"]
    return context


@dav2tf_registry.register_func(["add_n"])
def add_n(context: "tbetoolkits.UniversalTestcaseStructure"):
    """Davinci operators tend to use tensor as axis input, while tf ops don't"""
    if "tensor_num" in context.other_compilation_params:
        params = context.other_compilation_params
    elif "tensor_num" in context.other_runtime_params:
        params = context.other_runtime_params
    else:
        return context
    params = params.copy()
    tensor_num = params["tensor_num"]
    params["n"] = tensor_num
    del params["tensor_num"]
    return context


@dav2tf_registry.register_func(["tile"])
def tile(context: "tbetoolkits.UniversalTestcaseStructure"):
    """Davinci operator"""
    if "input_m" in context.other_compilation_params:
        params = context.other_compilation_params
    elif "input_m" in context.other_runtime_params:
        params = context.other_runtime_params
    else:
        return context
    if "input_m" in params and "multiples" not in params:
        params["multiples"] = params["input_m"]
        del params["input_m"]
    return context


@dav2tf_registry.register_func(["cast"])
def cast(context: "tbetoolkits.UniversalTestcaseStructure"):
    """Davinci Cast to tf cast"""
    if "dst_type" in context.other_compilation_params:
        params = context.other_compilation_params
    elif "dst_type" in context.other_runtime_params:
        params = context.other_runtime_params
    else:
        return context

    def _cast_dsttype_conversion(_dst_type):
        """
        This function is determined by TBE operator implementation
        Update is needed if TBE operator changed
        :param _dst_type: int
        :return: str
        """
        if _dst_type == 0:
            _dst_type = "float32"
        if _dst_type == 1:
            _dst_type = "float16"
        if _dst_type == 2:
            _dst_type = "int8"
        if _dst_type == 3:
            _dst_type = "int32"
        if _dst_type == 4:
            _dst_type = "uint8"
        if _dst_type == 5:
            _dst_type = "int64"
        if _dst_type == 10:
            _dst_type = "uint64"
        if _dst_type == 12:
            _dst_type = "bool"
        return _dst_type

    params["dtype"] = _cast_dsttype_conversion(params["dst_type"])
    del params["dst_type"]
    return context


@dav2tf_registry.register_func(["split_v"])
def split_v(context: "tbetoolkits.UniversalTestcaseStructure"):
    """Davinci split_v to tf split_v"""
    if "split_dim" in context.other_compilation_params:
        params = context.other_compilation_params
    elif "split_dim" in context.other_runtime_params:
        params = context.other_runtime_params
    else:
        return context

    params = params.copy()
    params["axis"] = params["split_dim"]
    del params["split_dim"]
    return context


@dav2tf_registry.register_func(["top_k_d"])
def split_v(context: "tbetoolkits.UniversalTestcaseStructure"):
    """Davinci split_v to tf split_v"""
    context.stc_ori_inputs = (context.stc_ori_inputs[0],)
    return context
