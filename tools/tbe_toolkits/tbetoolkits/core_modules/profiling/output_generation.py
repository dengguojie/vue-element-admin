#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
output generation method for Universal testcases
"""
# Standard Packages
import inspect
import logging
from typing import Callable, Sequence, Tuple

# Third-party Packages
import numpy
from ..testcase_manager import UniversalTestcaseStructure
from ...utilities import bfloat16_conversion
from ...utilities import get_global_storage
from ...utilities import get


def __gen_output(context: UniversalTestcaseStructure):
    from ...user_defined_modules.golden_funcs.registry import golden_funcs
    # Enable tensorflow numpy bfloat16 support
    output_dtypes = bfloat16_conversion(context.output_dtypes)
    # Following scenario will disable output generation sequence
    # 1. Input Array Missing
    # 2. Precision Test Disabled
    if not context.input_arrays:
        return
    if not get_global_storage().do_precision_test:
        context.golden_arrays = ("SUPPRESSED",)
        context.output_byte_arrays = tuple(numpy.zeros(shape, get(output_dtypes, idx)).tobytes()
                                           for idx, shape in enumerate(context.stc_outputs) if shape is not None)
        return
    other_runtime_params = context.other_runtime_params.copy()
    output_byte_arrays = []
    if context.manual_output_data_binaries:
        golden_arrays = [numpy.fromfile(file_path, dtype=get(output_dtypes, i))
                         for i, file_path in enumerate(context.manual_output_data_binaries)]
    else:
        golden_arrays = []
        if context.op_name not in golden_funcs:
            if context.op_name in numpy.__dir__() and isinstance(getattr(numpy, context.op_name), Callable):
                golden_funcs[context.op_name] = getattr(numpy, context.op_name)
        if context.op_name in golden_funcs:
            ################################
            # Golden Data Generation
            ################################
            # noinspection PyBroadException
            try:
                golden_parameters = inspect.signature(golden_funcs[context.op_name]).parameters
            except:
                golden_parameters = []
            # Parameter correction
            if "axes" in other_runtime_params:
                other_runtime_params["axis"] = other_runtime_params["axes"]
                del other_runtime_params["axes"]
            for key in tuple(other_runtime_params.keys()):
                if key not in golden_parameters:
                    del other_runtime_params[key]
            if "actual_formats" in golden_parameters:
                other_runtime_params["actual_formats"] = context.stc_input_formats
            if "ori_shapes" in golden_parameters:
                other_runtime_params["ori_shapes"] = context.stc_ori_inputs
            if "ori_formats" in golden_parameters:
                other_runtime_params["ori_formats"] = context.stc_input_ori_formats
            # Call golden function
            # noinspection PyBroadException
            try:
                if "context" in golden_parameters:
                    golden_results = golden_funcs[context.op_name](context)
                else:
                    golden_results = golden_funcs[context.op_name](*context.input_arrays, **other_runtime_params)
            except:
                logging.exception("Golden generation failure:")
                golden_results = ["GOLDEN_FAILURE" for _ in range(len(context.stc_outputs))]
            if isinstance(golden_results, Sequence):
                for golden_result in golden_results:
                    golden_arrays.append(golden_result)
            elif isinstance(golden_results, numpy.generic):
                golden_arrays.append(numpy.array((golden_results,)))
            else:
                golden_arrays.append(golden_results)
        else:
            logging.warning("Golden data generation method for operator %s is not registered!" % context.op_name)
            for i in range(len(context.stc_outputs)):
                golden_arrays.append("UNSUPPORTED")
    for idx, golden_array in enumerate(golden_arrays):
        # Enable inplace
        if idx < len(context.output_inplace_indexes) and context.output_inplace_indexes[idx] is not None:
            output_byte_arrays.append(context.output_inplace_indexes[idx])
            if isinstance(golden_arrays[idx], numpy.ndarray):
                golden_arrays[idx] = golden_arrays[idx].astype(get(output_dtypes, idx)).flatten()
        else:
            if isinstance(golden_arrays[idx], numpy.ndarray):
                if context.stc_outputs[idx] and tuple(context.stc_outputs[idx]) != tuple(golden_arrays[idx].shape):
                    logging.warning(f"Golden shape {golden_arrays[idx].shape} not match with testcase shape "
                                    f"{context.stc_outputs[idx]}, replacing...")
                golden_arrays[idx] = golden_arrays[idx].astype(get(output_dtypes, idx)).flatten()
                output_byte_arrays.append(b'\00' * golden_arrays[idx].nbytes)

            elif golden_arrays[idx] is None:
                pass
            else:
                output_byte_arrays.append(numpy.zeros(context.stc_outputs[idx], get(output_dtypes, idx)).tobytes())
    del context.input_arrays
    del context.dyn_input_arrays
    context.golden_arrays = tuple(golden_arrays)
    context.output_byte_arrays = tuple(output_byte_arrays)


def __gen_workspaces(workspaces: Tuple[int]) -> Tuple[bytes]:
    workspace_byte_arrays = []
    for workspace_size in workspaces:
        if isinstance(workspace_size, str):
            return tuple(workspace_byte_arrays)
        if workspace_size:
            workspace_byte_arrays.append(numpy.zeros((int(workspace_size),), dtype="uint8").tobytes())
    return tuple(workspace_byte_arrays)
