#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Input data range inference function for element_wise pattern
"""
# Standard Packages
import math
# Third-Party Packages
import numpy
from ...utilities import get


def _reduce_prod(range_mode,
                 idx,
                 dyn_inputs,
                 stc_inputs,
                 dyn_input_dtypes,
                 other_runtime_params):
    inputA_dtype = get(dyn_input_dtypes, 0)
    inputB_dtype = get(dyn_input_dtypes, 1)
    if inputA_dtype not in ("float16", "float32", "int8", "uint8", "int32"):
        raise RuntimeError("Operator ReduceProd first input x not support [%s]" % inputA_dtype)
    if inputB_dtype not in ("int32", "int64"):
        raise RuntimeError("Operator ReduceProd second input axes not support [%s]" % inputB_dtype)
    reduce_axis = other_runtime_params.get("axes")
    if reduce_axis is None:
        reduce_axis = other_runtime_params.get("axis")
    reduce_axis = list(reduce_axis)
    shape_x = list(stc_inputs[0])
    shape_reduce = 1
    for i in range(0, len(reduce_axis)):
        shape_reduce *= shape_x[reduce_axis[i]]
    numpy_dtype = numpy.dtype(inputA_dtype)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max

    if range_mode == "LOW":
        if idx == 0:
            if inputA_dtype in ("int8", "uint8"):
                return 0
            else:
                return 1.0
        elif idx == 1:
            return -len(dyn_inputs[1])
        else:
            raise RuntimeError("Operator ReduceProd should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            return int(math.pow(math.fabs(maximum), float(1/shape_reduce)))
        elif idx == 1:
            return len(dyn_inputs[1]) - 1
        else:
            raise RuntimeError("Operator ReduceProd should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _reduce_sum(range_mode,
                idx,
                dyn_inputs,
                stc_inputs,
                dyn_input_dtypes,
                other_runtime_params):
    inputA_dtype = get(dyn_input_dtypes, 0)
    inputB_dtype = get(dyn_input_dtypes, 1)
    if inputA_dtype not in ("float16", "float32", "int32"):
        raise RuntimeError("Operator ReduceSum first input x not support [%s]" % inputA_dtype)
    if inputB_dtype not in ("int32", "int64"):
        raise RuntimeError("Operator ReduceSum second input axes not support [%s]" % inputB_dtype)
    reduce_axis = other_runtime_params.get("axes")
    if reduce_axis is None:
        reduce_axis = other_runtime_params.get("axis")
    reduce_axis = list(reduce_axis)
    shape_x = list(stc_inputs[0])
    shape_reduce = 1
    for i in range(0, len(reduce_axis)):
        shape_reduce *= shape_x[reduce_axis[i]]
    numpy_dtype = numpy.dtype(inputA_dtype)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max

    if range_mode == "LOW":
        if idx == 0:
            return 0
        elif idx == 1:
            return -len(dyn_inputs[1])
        else:
            raise RuntimeError("Operator ReduceSum should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            if inputA_dtype in ("float16",):
                return math.pow(math.fabs(maximum) // shape_reduce, 0.5)
            else:
                return math.pow(math.fabs(maximum) // shape_reduce, 0.25)
        elif idx == 1:
            return len(dyn_inputs[1]) - 1
        else:
            raise RuntimeError("Operator ReduceSum should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _reduce_sum_d(range_mode,
                  idx,
                  dyn_inputs,
                  stc_inputs,
                  dyn_input_dtypes,
                  other_runtime_params):
    [dyn_inputs].clear()
    input_dtype = get(dyn_input_dtypes, 0)
    if input_dtype not in ("float16", "float32", "int32"):
        raise RuntimeError("Operator ReduceSumD first input x not support [%s]" % input_dtype)
    reduce_axis = other_runtime_params.get("axes")
    if reduce_axis is None:
        reduce_axis = other_runtime_params.get("axis")
    reduce_axis = list(reduce_axis)
    shape_x = list(stc_inputs[0])
    shape_reduce = 1
    for i in range(0, len(reduce_axis)):
        shape_reduce *= shape_x[reduce_axis[i]]
    
    if shape_reduce == 0:
        return 0
    numpy_dtype = numpy.dtype(input_dtype)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max

    if range_mode == "LOW":
        if idx == 0:
            return 0
        else:
            raise RuntimeError("Operator ReduceSumD should have one input only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            if input_dtype in ("float16",):
                return math.pow(math.fabs(maximum) // shape_reduce, 0.5)
            else:
                return math.pow(math.fabs(maximum) // shape_reduce, 0.25)
        else:
            raise RuntimeError("Operator ReduceSumD should have one input only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _bias_add_grad(range_mode,
                   idx,
                   dyn_inputs,
                   stc_inputs,
                   dyn_input_dtypes,
                   other_runtime_params):
    input_dtype = get(dyn_input_dtypes, 0)
    if input_dtype not in ("float16", "float32"):
        raise RuntimeError("Operator BiasAddGrad first input x not support [%s]" % input_dtype)
    [dyn_inputs, other_runtime_params].clear()
    shape_x = list(stc_inputs[0])
    shape_reduce = 1
    for i in range(0, len(shape_x)):
        shape_reduce *= shape_x[i]
    numpy_dtype = numpy.dtype(input_dtype)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max

    if range_mode == "LOW":
        if idx == 0:
            return 0
        else:
            raise RuntimeError("Operator BiasAddGrad should have one input only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            if input_dtype in ("float16",):
                return math.pow(math.fabs(maximum) // shape_reduce, 0.5)
            else:
                return math.pow(math.fabs(maximum) // shape_reduce, 0.25)
        else:
            raise RuntimeError("Operator BiasAddGrad should have one input only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _reduce_mean_d(range_mode,
                   idx,
                   dyn_inputs,
                   stc_inputs,
                   dyn_input_dtypes,
                   other_runtime_params):
    [dyn_inputs].clear()
    input_dtype = get(dyn_input_dtypes, 0)
    if input_dtype not in ("float16", "float32", "uint8", "int8"):
        raise RuntimeError("Operator ReduceMeanD first input x not support [%s]" % input_dtype)
    reduce_axis = other_runtime_params.get("axes")
    if reduce_axis is None:
        reduce_axis = other_runtime_params.get("axis")
    reduce_axis = list(reduce_axis)
    shape_x = list(stc_inputs[0])
    shape_reduce = 1
    for i in range(0, len(reduce_axis)):
        shape_reduce *= shape_x[reduce_axis[i]]
    numpy_dtype = numpy.dtype(input_dtype)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max

    if range_mode == "LOW":
        if idx == 0:
            return 0
        else:
            raise RuntimeError("Operator ReduceMeanD should have one input only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            return math.fabs(maximum) // shape_reduce
        else:
            raise RuntimeError("Operator ReduceMeanD should have one input only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)
