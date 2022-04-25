#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Input data range inference function for element_wise pattern
"""
# Standard Packages
import math

# Third-Party Packages
import numpy
from .utilities import input_check
from ...utilities import get


def _add(range_mode,
         idx,
         dyn_inputs,
         stc_inputs,
         dyn_input_dtypes,
         other_runtime_params):
    # Use them once to avoid static checks
    [idx, stc_inputs, other_runtime_params].clear()
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator Add should have two inputs with the same dtype!!!")
    numpy_dtype = numpy.dtype(dyn_input_dtypes[0])
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        return minimum / 3
    elif range_mode == "HIGH":
        return maximum / 3
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _sub(range_mode,
         idx,
         dyn_inputs,
         stc_inputs,
         dyn_input_dtypes,
         other_runtime_params):
    # Use them once to avoid static checks
    [idx, stc_inputs, other_runtime_params].clear()
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator Sub should have two inputs with the same dtype!!!")
    numpy_dtype = numpy.dtype(dyn_input_dtypes[0])
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        return minimum / 3
    elif range_mode == "HIGH":
        return maximum / 3
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _div(range_mode,
         idx,
         dyn_inputs,
         stc_inputs,
         dyn_input_dtypes,
         other_runtime_params):
    # Use them once to avoid static checks
    [stc_inputs, other_runtime_params].clear()
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator Div should have two inputs with the same dtype!!!")
    numpy_dtype = numpy.dtype(dyn_input_dtypes[0])
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        if idx == 0:
            return minimum
        elif idx == 1:
            return 1
        else:
            raise RuntimeError("Operator Div should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            return maximum
        elif idx == 1:
            return maximum / 2
        else:
            raise RuntimeError("Operator Div should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _maximum(range_mode,
             idx,
             dyn_inputs,
             stc_inputs,
             dyn_input_dtypes,
             other_runtime_params):
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator Maximum should have two inputs with the same dtype!!!")
    inputA_dtype = get(dyn_input_dtypes, 0)
    # Use them once to avoid static checks
    [stc_inputs, other_runtime_params].clear()
    numpy_dtype = numpy.dtype(inputA_dtype)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        if idx in (0, 1):
            return minimum
        else:
            raise RuntimeError("Operator Maximum should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx in (0, 1):
            return maximum
        else:
            raise RuntimeError("Operator Maximum should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _minimum(range_mode,
             idx,
             dyn_inputs,
             stc_inputs,
             dyn_input_dtypes,
             other_runtime_params):
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator Minimum should have two inputs with the same dtype!!!")
    inputA_dtype = get(dyn_input_dtypes, 0)
    # Use them once to avoid static checks
    [stc_inputs, other_runtime_params].clear()
    numpy_dtype = numpy.dtype(inputA_dtype)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        if idx in (0, 1):
            return minimum
        else:
            raise RuntimeError("Operator Minimum should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx in (0, 1):
            return maximum
        else:
            raise RuntimeError("Operator Minimum should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _mul(range_mode,
         idx,
         dyn_inputs,
         stc_inputs,
         dyn_input_dtypes,
         other_runtime_params):
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator Mul should have two inputs with the same dtype!!!")
    inputA_dtype = get(dyn_input_dtypes, 0)
    # Use them once to avoid static checks
    [stc_inputs, other_runtime_params].clear()
    numpy_dtype = numpy.dtype(inputA_dtype)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        if idx == 0:
            return math.sqrt(minimum) if minimum > 0 else -math.sqrt(-minimum)
        elif idx == 1:
            return math.sqrt(minimum) if minimum > 0 else -math.sqrt(-minimum)
        else:
            raise RuntimeError("Operator Mul should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            return math.sqrt(maximum) if maximum > 0 else -math.sqrt(-maximum)
        elif idx == 1:
            return math.sqrt(maximum) if maximum > 0 else -math.sqrt(-maximum)
        else:
            raise RuntimeError("Operator Mul should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _floor_div(range_mode,
               idx,
               dyn_inputs,
               stc_inputs,
               dyn_input_dtypes,
               other_runtime_params):
    # Use them once to avoid static checks
    [stc_inputs, other_runtime_params].clear()
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator FloorDiv should have two inputs with the same dtype!!!")
    numpy_dtype = numpy.dtype(dyn_input_dtypes[0])
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        if idx == 0:
            return minimum
        elif idx == 1:
            return 1
        else:
            raise RuntimeError("Operator FloorDiv should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            return maximum
        elif idx == 1:
            return maximum / 2
        else:
            raise RuntimeError("Operator FloorDiv should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _real_div(range_mode,
              idx,
              dyn_inputs,
              stc_inputs,
              dyn_input_dtypes,
              other_runtime_params):
    # Use them once to avoid static checks
    [stc_inputs, other_runtime_params].clear()
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator RealDiv should have two inputs with the same dtype!!!")
    numpy_dtype = numpy.dtype(dyn_input_dtypes[0])
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        if idx == 0:
            return minimum
        elif idx == 1:
            return 1
        else:
            raise RuntimeError("Operator RealDiv should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            return maximum
        elif idx == 1:
            return maximum / 2
        else:
            raise RuntimeError("Operator RealDiv should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _mod(range_mode,
         idx,
         dyn_inputs,
         stc_inputs,
         dyn_input_dtypes,
         other_runtime_params):
    # Use them once to avoid static checks
    [stc_inputs, other_runtime_params].clear()
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator Mod should have two inputs with the same dtype!!!")
    numpy_dtype = numpy.dtype(dyn_input_dtypes[0])
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        if idx == 0:
            return minimum
        elif idx == 1:
            return 1
        else:
            raise RuntimeError("Operator Mod should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            return maximum
        elif idx == 1:
            return maximum / 2
        else:
            raise RuntimeError("Operator Mod should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _floor_mod(range_mode,
               idx,
               dyn_inputs,
               stc_inputs,
               dyn_input_dtypes,
               other_runtime_params):
    # Use them once to avoid static checks
    [stc_inputs, other_runtime_params].clear()
    if not input_check(dyn_inputs, dyn_input_dtypes, 2):
        raise RuntimeError("Operator FloorMod should have two inputs with the same dtype!!!")
    numpy_dtype = numpy.dtype(dyn_input_dtypes[0])
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min
    if range_mode == "LOW":
        if idx == 0:
            return minimum
        elif idx == 1:
            return 1
        else:
            raise RuntimeError("Operator FloorMod should have two inputs only!!!")
    elif range_mode == "HIGH":
        if idx == 0:
            return maximum
        elif idx == 1:
            return maximum / 2
        else:
            raise RuntimeError("Operator FloorMod should have two inputs only!!!")
    else:
        raise KeyError("Unknown range inference mode %s" % range_mode)


def _relu_grad_v2(range_mode,
                  idx,
                  dyn_inputs,
                  stc_inputs,
                  dyn_input_dtypes,
                  other_runtime_params):
    # Use them once to avoid static checks
    [stc_inputs, dyn_inputs, other_runtime_params].clear()
    numpy_dtype = numpy.dtype(dyn_input_dtypes[0])
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    maximum = numpy_info.max
    minimum = numpy_info.min

    if range_mode not in ("LOW", "HIGH"):
        raise KeyError("Unknown range inference mode %s" % range_mode)

    if idx not in (0, 1):
        raise RuntimeError("Operator ReluGradV2 should have two inputs only !!!")

    if range_mode == "LOW":
        return minimum if idx == 0 else 0
    else:
        return maximum if idx == 0 else 1.99
