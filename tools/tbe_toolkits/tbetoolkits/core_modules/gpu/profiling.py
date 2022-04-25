#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Profiling method for Universal testcases
"""
# Standard Packages
import logging
import multiprocessing

# Third-party Packages

from .pytorch_profiling import pytorch_profiling
from .tensorflow_profiling import tensorflow_profiling
from ..tbe_multiprocessing.pool import get_process_context
from ...utilities import get
from ...utilities import MODE
from ...utilities import shape_product
from ...utilities import get_dtype_width
from ...utilities import get_global_storage
from ...utilities import set_thread_name
from ...utilities import set_process_name
from ..testcase_manager import UniversalTestcaseStructure


def __notify_status(status: str):
    set_thread_name(status)
    get_process_context().send_data("stage", status)


def __report_process_name(my_name_is: str):
    set_process_name(my_name_is)
    get_process_context().change_name(my_name_is)


# noinspection DuplicatedCode
def profile_process(context: UniversalTestcaseStructure, device_id):
    """
    Universal Testcase Profiling Entrance
    """
    __report_process_name(context.testcase_name)
    multiprocessing.current_process().name = context.testcase_name
    __notify_status("OnTestcaseProfilingStart")
    total_memory_size = sum((sum(tuple(shape_product(context.stc_inputs[idx])
                                       * get_dtype_width(get(context.stc_input_dtypes, idx))
                                       for idx in range(len(context.stc_inputs))
                                       if context.stc_inputs[idx] is not None)),
                             sum(tuple(shape_product(context.stc_outputs[idx])
                                       * get_dtype_width(get(context.output_dtypes, idx))
                                       for idx in range(len(context.stc_outputs))
                                       if context.stc_outputs[idx] is not None))))
    if get_global_storage().mode == MODE.GPU_TENSORFLOW:
        perf = tensorflow_profiling(context, device_id)
    else:
        perf = pytorch_profiling(context, device_id)
    if not isinstance(perf, str):
        throughput = str(total_memory_size / 1024 / 1024 / 1024 / (perf / 1000 / 1000)) if perf > 0 else str(0)
    else:
        throughput = perf
    logging.info("\n===========================\n"
                 f"Perf Mode: {get_global_storage().mode.name}\n"
                 f"GPU Index: {device_id}\n"
                 f"GPU time: {perf} us\n"
                 f"GPU throughput: {throughput} GB/s\n"
                 "===========================\n")
    return perf, throughput
