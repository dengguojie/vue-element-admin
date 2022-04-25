#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Input generation method for Universal testcases
"""
# Standard Packages
import inspect
import logging
from typing import List, Optional

# Third-party Packages
import numpy
from ..testcase_manager import UniversalTestcaseStructure
from ...utilities import bfloat16_conversion
from ...utilities import eliminate_scalar_shapes
from ...utilities import get

default_low = -2
default_high = 2


def __gen_input(context: UniversalTestcaseStructure):
    from ...user_defined_modules.input_funcs.registry import special_input_func
    from ...user_defined_modules.input_range_funcs.registry import input_range_func
    # Enable bfloat16 support
    dyn_input_dtypes = bfloat16_conversion(context.dyn_input_dtypes)
    stc_input_dtypes = bfloat16_conversion(context.stc_input_dtypes)
    # Manual inputs
    if context.manual_input_data_binaries:
        dyn_input_arrays = context.manual_input_data_binaries[0]
        stc_input_arrays = context.manual_input_data_binaries[1]
        ################################
        # Read from Disk
        ################################
        # Actual input data ranges are Unknown
        actual_input_data_ranges = ((None, None),)
        # Use _stc_ to declare static shape input
        stc_input_arrays = tuple(numpy.fromfile(file_path, dtype=get(stc_input_dtypes, i))
                                 for i, file_path in enumerate(stc_input_arrays))
        stc_input_byte_arrays = [array.tobytes() for array in stc_input_arrays]
        # Use _dyn_ to declare dynamic shape input
        dyn_input_arrays = tuple(numpy.fromfile(file_path, dtype=get(dyn_input_dtypes, i))
                                 for i, file_path in enumerate(dyn_input_arrays))
        dyn_input_byte_arrays = [array.tobytes() for array in dyn_input_arrays]
        context.input_arrays = tuple(stc_input_arrays)
        context.stc_input_byte_arrays = tuple(stc_input_byte_arrays)
        context.dyn_input_arrays = tuple(dyn_input_arrays)
        context.dyn_input_byte_arrays = tuple(dyn_input_byte_arrays)
        context.actual_input_data_ranges = tuple(actual_input_data_ranges)
        return
    # Automatic input generation
    other_runtime_params: dict = context.other_runtime_params.copy()
    input_arrays: List[Optional[numpy.ndarray]] = []
    stc_input_byte_arrays: List[bytes] = []
    dyn_input_byte_arrays: List[bytes] = []
    # Translate input params
    dyn_inputs = eliminate_scalar_shapes(context.dyn_inputs)
    stc_inputs = eliminate_scalar_shapes(context.stc_inputs)
    ################################
    # Realtime Input Data Generation (Default)
    ################################
    warning_switch = True
    actual_input_data_ranges = []
    for idx, input_data_shape in enumerate(stc_inputs):
        if input_data_shape is None:
            input_arrays.append(None)
            actual_input_data_ranges.append((None, None))
            continue
        if context.op_name not in input_range_func and warning_switch:
            logging.warning("Unable to do automatic input range inference for operator %s, using default value"
                            % context.op_name)
            warning_switch = False
        low = get(context.input_data_ranges, idx)[0]
        high = get(context.input_data_ranges, idx)[1]
        # noinspection PyBroadException
        try:
            if low is None:
                if context.op_name in input_range_func:
                    low = input_range_func[context.op_name]("LOW",
                                                            idx,
                                                            dyn_inputs,
                                                            stc_inputs,
                                                            dyn_input_dtypes,
                                                            other_runtime_params)
                else:
                    low = default_low
            if high is None:
                if context.op_name in input_range_func:
                    high = input_range_func[context.op_name]("HIGH",
                                                             idx,
                                                             dyn_inputs,
                                                             stc_inputs,
                                                             dyn_input_dtypes,
                                                             other_runtime_params)
                else:
                    high = default_high
        except:
            logging.exception("Special input range function failure:")
            low = default_low
            high = default_high
        actual_input_data_ranges.append((low, high))
        input_arrays.append(numpy.random.uniform(low, high,
                                                 input_data_shape).astype(get(stc_input_dtypes, idx)))
    # Check special input operators
    golden_input_arrays = None
    if context.op_name in special_input_func:
        input_parameters = inspect.signature(special_input_func[context.op_name]).parameters
        if "ori_shapes" in input_parameters:
            other_runtime_params["ori_shapes"] = context.stc_ori_inputs
        if "ori_formats" in input_parameters:
            other_runtime_params["ori_formats"] = context.stc_input_ori_formats
        if "context" in input_parameters:
            context.original_input_arrays = input_arrays.copy()
            context.input_arrays = input_arrays
            context.actual_input_data_ranges = tuple(actual_input_data_ranges)
            special_return = special_input_func[context.op_name](context)
        else:
            context.original_input_arrays = input_arrays.copy()
            special_return = special_input_func[context.op_name](dyn_inputs,
                                                                 dyn_input_dtypes,
                                                                 input_arrays,
                                                                 other_runtime_params,
                                                                 actual_input_data_ranges,
                                                                 actual_input_data_ranges)
        if len(special_return) == 2:
            dyn_input_arrays, stc_input_arrays = special_return
        elif len(special_return) == 3:
            dyn_input_arrays, stc_input_arrays, golden_input_arrays = special_return
        else:
            raise RuntimeError("Special input function of operator %s returns invalid number of output %d"
                               % (context.op_name, len(special_return)))
    elif len(context.const_input_indexes) > 0 \
            and (context.dyn_compile_result == "SUCC"
                 or context.cst_compile_result == "SUCC"
                 or context.bin_compile_result == "SUCC"):
        if len(dyn_inputs) <= len(input_arrays):
            logging.warning("inputs abnormal, check length of stc_inputs and dyn_inputs or test may fail,")
            for const_input_index in context.const_input_indexes:
                del input_arrays[const_input_index]
        dyn_input_arrays = input_arrays[:]
        stc_input_arrays = input_arrays
        for idx, const_input_index in enumerate(context.const_input_indexes):
            const_input_param_name = context.dyn_func_params[const_input_index] \
                if get(context.const_input_modes, idx) is None else get(context.const_input_modes, idx)
            const_input_param_dtype = get(dyn_input_dtypes, const_input_index)
            # noinspection PyBroadException
            try:
                const_input_param_value = other_runtime_params[const_input_param_name]
                const_input_array = numpy.array(const_input_param_value, dtype=const_input_param_dtype)
                dyn_input_arrays.insert(const_input_index, const_input_array)
            except:
                if const_input_param_name == "axis" and "axes" in other_runtime_params:
                    const_input_param_value = other_runtime_params["axes"]
                    const_input_array = numpy.array(const_input_param_value, dtype=const_input_param_dtype)
                    dyn_input_arrays.insert(const_input_index, const_input_array)
                else:
                    raise
    else:
        dyn_input_arrays = input_arrays
        stc_input_arrays = input_arrays
    for array in stc_input_arrays:
        if array is not None:
            stc_input_byte_arrays.append(array.tobytes())
    if context.op_name in special_input_func or len(context.const_input_indexes) > 0:
        for array in dyn_input_arrays:
            if array is not None:
                dyn_input_byte_arrays.append(array.tobytes())
    else:
        dyn_input_byte_arrays = stc_input_byte_arrays
    context.input_arrays = tuple(stc_input_arrays)
    context.stc_input_byte_arrays = tuple(stc_input_byte_arrays)
    context.dyn_input_arrays = tuple(dyn_input_arrays)
    context.dyn_input_byte_arrays = tuple(dyn_input_byte_arrays)
    context.actual_input_data_ranges = tuple(actual_input_data_ranges)
    if golden_input_arrays:
        logging.debug("Using special input arrays for golden generation function of operator %s" % context.op_name)
        context.input_arrays = tuple(golden_input_arrays)
