#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Profiling method for Universal testcases
"""
# Standard Packages
import os
import gc
import math
import time
import logging
import pathlib
import shutil
import platform as plat
import multiprocessing.synchronize
from typing import NoReturn
from typing import Optional
from typing import Tuple

try:
    from contextlib import nullcontext
except ImportError:
    print("Python version too low, from contextlib import nullcontext failed")
    import contextlib


    @contextlib.contextmanager
    def NULLCXT():
        """NULL CONTEXT"""
        pass


    nullcontext = NULLCXT

# Third-Party Packages
import tbetoolkits
import numpy
import psutil
from .comparison import comparing
from .input_generation import __gen_input
from .output_generation import __gen_output
from .output_generation import __gen_workspaces
from .rts_sequence import rtsProfilingCallEx
from ..runtime import RTSInterface
from ..tbe_logging import default_logging_config
from ..tbe_multiprocessing.pool import get_process_context
from ..testcase_manager import UniversalTestcaseStructure
from ...utilities import get
from ...utilities import shape_product
from ...utilities import get_dtype_width
from ...utilities import set_thread_name
from ...utilities import set_process_name
from ...utilities import get_global_storage
from ...utilities import parse_tiling_data
from ...utilities import get_str_tiling_data


class RTSProfilingResult:
    """
    RTS Profiling output
    """

    def __init__(self, cycle=None, output_bytes=(None,), pmu=()):
        self.cycle: Optional[str] = cycle
        self.output_bytes: Optional[tuple] = output_bytes
        self.pmu: Optional[tuple] = pmu


class ProfilingReturnStructure:
    """
    Structure for Return
    """

    __slots__ = ("tiling_time_us",
                 "dyn_name",
                 "stc_name",
                 "cst_name",
                 "bin_name",
                 "dyn_block_dim",
                 "stc_block_dim",
                 "cst_block_dim",
                 "bin_block_dim",
                 "dyn_perf_us",
                 "stc_perf_us",
                 "cst_perf_us",
                 "bin_perf_us",
                 "dyn_throughput_gbs",
                 "stc_throughput_gbs",
                 "cst_throughput_gbs",
                 "bin_throughput_gbs",
                 "dyn_compile_s",
                 "stc_compile_s",
                 "cst_compile_s",
                 "bin_compile_s",
                 "perf_status",
                 "cst_perf_status",
                 "dyn_precision",
                 "stc_precision",
                 "rel_precision",
                 "cst_precision",
                 "rst_precision",
                 "bin_precision",
                 "bre_precision",
                 "precision_status",
                 "data_input_size_b",
                 "data_output_size_b",
                 "dyn_sch_count",
                 "dyn_obj_size_b",
                 "dyn_kernel_size_b",
                 "dyn_tiling_data",
                 "dyn_tiling_key",
                 "dyn_op_pattern",
                 "cst_op_pattern",
                 "stc_rl_query_result",
                 "cst_rl_query_result",
                 "stc_op_pattern",
                 "dyn_pmu",
                 "stc_pmu",
                 "cst_pmu")

    def __init__(self, default_value=None):
        self.tiling_time_us = default_value
        # DYN
        self.dyn_name = default_value
        self.dyn_block_dim = default_value
        self.dyn_perf_us = default_value
        self.dyn_throughput_gbs = default_value
        self.dyn_compile_s = default_value
        self.dyn_precision = default_value
        self.dyn_pmu = ("UNKNOWN",) * get_global_storage().PMU_RETURN_SIZE if get_global_storage().PMU else ()
        # STC
        self.stc_name = default_value
        self.stc_block_dim = default_value
        self.stc_perf_us = default_value
        self.stc_throughput_gbs = default_value
        self.stc_compile_s = default_value
        self.stc_precision = default_value
        self.stc_rl_query_result = default_value
        self.cst_rl_query_result = default_value
        self.stc_op_pattern = default_value
        self.stc_pmu = ("UNKNOWN",) * get_global_storage().PMU_RETURN_SIZE if get_global_storage().PMU else ()
        # CST
        self.cst_name = default_value
        self.cst_block_dim = default_value
        self.cst_perf_us = default_value
        self.cst_throughput_gbs = default_value
        self.cst_compile_s = default_value
        self.cst_precision = default_value
        self.rst_precision = default_value
        self.cst_pmu = ("UNKNOWN",) * get_global_storage().PMU_RETURN_SIZE if get_global_storage().PMU else ()
        # DYN
        self.bin_name = default_value
        self.bin_block_dim = default_value
        self.bin_perf_us = default_value
        self.bin_throughput_gbs = default_value
        self.bin_compile_s = default_value
        self.bin_precision = default_value
        self.bre_precision = default_value

        self.perf_status = default_value
        self.cst_perf_status = default_value
        # Precision
        self.rel_precision = default_value
        self.precision_status = default_value
        # Special
        self.data_input_size_b = default_value
        self.data_output_size_b = default_value
        self.dyn_sch_count = default_value
        self.dyn_obj_size_b = default_value
        self.dyn_kernel_size_b = default_value
        self.dyn_tiling_data = default_value
        self.dyn_tiling_key = default_value
        self.dyn_op_pattern = default_value
        self.cst_op_pattern = default_value

    # noinspection DuplicatedCode
    def construct(self, context: "", compare_result, passed, cst_passed, input_size: int, output_size: int):
        """Construct the structure with context"""
        # Check prof_results and construct one if necessary
        total_size = input_size + output_size
        if not isinstance(context.dyn_prof_result, RTSProfilingResult):
            context.dyn_prof_result = RTSProfilingResult(passed,
                                                         None,
                                                         self.dyn_pmu)
        if not isinstance(context.stc_prof_result, RTSProfilingResult):
            context.stc_prof_result = RTSProfilingResult(passed,
                                                         None,
                                                         self.stc_pmu)
        if not isinstance(context.cst_prof_result, RTSProfilingResult):
            context.cst_prof_result = RTSProfilingResult(passed,
                                                         None,
                                                         self.cst_pmu)
        if not isinstance(context.bin_prof_result, RTSProfilingResult):
            context.bin_prof_result = RTSProfilingResult(passed,
                                                         None,
                                                         self.dyn_pmu)
        self.tiling_time_us = context.dyn_tiling_time
        # DYN
        self.dyn_name = context.dyn_kernel_name
        self.dyn_block_dim = str(context.dyn_block_dim) \
            if context.dyn_block_dim != 0 else context.dyn_prof_result.cycle
        self.dyn_perf_us = context.dyn_prof_result.cycle
        # noinspection PyBroadException
        try:
            self.dyn_throughput_gbs = total_size / float(
                context.dyn_prof_result.cycle) / 1024 / 1024 / 1024 * 1000 * 1000
        except:
            self.dyn_throughput_gbs = self.dyn_perf_us
        self.dyn_pmu = context.dyn_prof_result.pmu if get_global_storage().PMU else ()
        self.dyn_compile_s = context.dyn_compile_time
        self.dyn_precision = compare_result[0]
        # STC
        self.stc_name = context.stc_kernel_name
        self.stc_block_dim = str(context.stc_block_dim) \
            if context.stc_block_dim != 0 else context.stc_prof_result.cycle
        self.stc_perf_us = context.stc_prof_result.cycle
        # noinspection PyBroadException
        try:
            self.stc_throughput_gbs = total_size / float(
                context.stc_prof_result.cycle) / 1024 / 1024 / 1024 * 1000 * 1000
        except:
            self.stc_throughput_gbs = self.stc_perf_us
        self.stc_compile_s = context.stc_compile_time
        self.stc_precision = compare_result[1]
        self.stc_pmu = context.stc_prof_result.pmu if get_global_storage().PMU else ()
        # CST
        self.cst_name = context.cst_kernel_name
        self.cst_block_dim = str(context.cst_block_dim) \
            if context.cst_block_dim != 0 else context.cst_prof_result.cycle
        self.cst_perf_us = context.cst_prof_result.cycle
        # noinspection PyBroadException
        try:
            self.cst_throughput_gbs = total_size / float(
                context.cst_prof_result.cycle) / 1024 / 1024 / 1024 * 1000 * 1000
        except:
            self.cst_throughput_gbs = self.cst_perf_us
        self.cst_compile_s = context.cst_compile_time
        self.cst_precision = compare_result[3]
        self.cst_pmu = context.cst_prof_result.pmu if get_global_storage().PMU else ()
        self.rst_precision = compare_result[4]
        # DYN
        self.bin_name = context.bin_kernel_name
        self.bin_block_dim = str(context.bin_block_dim) \
            if context.bin_block_dim != 0 else context.bin_prof_result.cycle
        self.bin_perf_us = context.bin_prof_result.cycle
        # noinspection PyBroadException
        try:
            self.bin_throughput_gbs = total_size / float(
                context.bin_prof_result.cycle) / 1024 / 1024 / 1024 * 1000 * 1000
        except:
            self.bin_throughput_gbs = self.bin_perf_us
        self.bin_compile_s = context.bin_compile_time
        self.bin_precision = compare_result[5]
        self.bre_precision = compare_result[6]
        self.perf_status = passed
        self.cst_perf_status = passed
        self.rel_precision = compare_result[2]
        self.precision_status = compare_result[7]
        self.data_input_size_b = input_size
        self.data_output_size_b = output_size
        self.dyn_sch_count = context.dyn_sch_count
        self.dyn_obj_size_b = context.dyn_obj_size
        self.dyn_kernel_size_b = context.dyn_kernel_size
        self.dyn_tiling_data = context.dyn_str_tiling_data
        self.dyn_tiling_key = context.dyn_tiling_key
        self.stc_rl_query_result = context.stc_rl_query_result
        self.cst_rl_query_result = context.cst_rl_status
        self.stc_op_pattern = context.stc_op_pattern
        self.dyn_op_pattern = context.dyn_op_pattern
        self.cst_op_pattern = context.cst_op_pattern

    def get(self):
        """
        Convert Structure to csv writable structure
        :return:
        """
        return (self.tiling_time_us,
                self.dyn_name,
                self.stc_name,
                self.cst_name,
                self.bin_name,
                self.dyn_block_dim,
                self.stc_block_dim,
                self.cst_block_dim,
                self.bin_block_dim,
                self.dyn_perf_us,
                self.stc_perf_us,
                self.cst_perf_us,
                self.bin_perf_us,
                self.dyn_throughput_gbs,
                self.stc_throughput_gbs,
                self.cst_throughput_gbs,
                self.bin_throughput_gbs,
                self.dyn_compile_s,
                self.stc_compile_s,
                self.cst_compile_s,
                self.bin_compile_s,
                self.perf_status,
                self.cst_perf_status,
                self.dyn_precision,
                self.stc_precision,
                self.rel_precision,
                self.cst_precision,
                self.rst_precision,
                self.bin_precision,
                self.bre_precision,
                self.precision_status,
                self.data_input_size_b,
                self.data_output_size_b,
                self.dyn_sch_count,
                self.dyn_obj_size_b,
                self.dyn_kernel_size_b,
                self.dyn_tiling_data,
                self.dyn_tiling_key,
                self.dyn_op_pattern,
                self.cst_op_pattern,
                self.stc_rl_query_result,
                self.cst_rl_query_result,
                self.stc_op_pattern,
                *self.dyn_pmu,
                *self.stc_pmu,
                *self.cst_pmu)


def profile_process(context: UniversalTestcaseStructure,
                    subprocess_device_locks: "Tuple[multiprocessing.synchronize.Lock]",
                    dev_id: int) -> ProfilingReturnStructure:
    """
    Universal Testcase Profiling Entrance
    """

    def __notify_status(status: str):
        set_thread_name(status)
        get_process_context().send_data("stage", status)

    def __report_process_name(my_name_is: str):
        set_process_name(my_name_is)
        get_process_context().change_name(my_name_is)

    __report_process_name(context.testcase_name)
    if get_global_storage().single_testcase_log_mode:
        default_logging_config(file_handler=get_global_storage().logging_to_file, testcase_name=context.testcase_name)
    ####################
    # Check whether there is need to do further test
    ####################
    if not context.is_valid:
        prof_result = ProfilingReturnStructure(context.fail_reason)
        if get_global_storage().PMU:
            prof_result.dyn_pmu = prof_result.stc_pmu = ("UNKNOWN",) * get_global_storage().PMU_RETURN_SIZE
        return prof_result
    __notify_status("OnParseParameters")
    __parse_manual_params(context)
    if context.dyn_is_valid:
        __parse_dynamic_tiling_data(context)
        __parse_binary_tiling_data(context)
    __notify_status("OnWaitingForMemory")
    __waiting_for_memory()
    input_memory_size = sum(tuple(shape_product(context.stc_inputs[idx])
                                  * get_dtype_width(get(context.stc_input_dtypes, idx))
                                  for idx in range(len(context.stc_inputs))
                                  if context.stc_inputs[idx] is not None))
    output_memory_size = sum(tuple(shape_product(context.stc_outputs[idx])
                                   * get_dtype_width(get(context.output_dtypes, idx))
                                   for idx in range(len(context.stc_outputs))
                                   if context.stc_outputs[idx] is not None))
    total_memory_size = input_memory_size + output_memory_size
    logging.debug(f"Expecting {total_memory_size} memory usage")
    __notify_status("OnGenInput")
    # noinspection PyBroadException
    try:
        __gen_input(context)
    except:
        logging.exception("Input data generation failure:")
        compare_result = ("INPUT_GEN_FAILURE",) * 8
        passed = "INPUT_GEN_FAILURE"
        return_structure = ProfilingReturnStructure()
        return_structure.construct(context, compare_result, passed, passed, input_memory_size, output_memory_size)
        __profiling_end_print(context, compare_result, passed)
        return return_structure
    __notify_status("OnGenGolden")
    # noinspection PyBroadException
    try:
        __gen_output(context)
    except:
        logging.exception("Output buffer initialization data or golden data generation failure:")
        compare_result = ("OUTPUT_GEN_FAILURE",) * 8
        passed = "OUTPUT_GEN_FAILURE"
        return_structure = ProfilingReturnStructure()
        return_structure.construct(context, compare_result, passed, passed, input_memory_size, output_memory_size)
        __profiling_end_print(context, compare_result, passed)
        return return_structure
    __notify_status("OnGenWorkspace")
    context.dyn_workspace_byte_arrays = __gen_workspaces(context.dyn_workspaces)
    context.stc_workspace_byte_arrays = __gen_workspaces(context.stc_workspaces)
    context.cst_workspace_byte_arrays = __gen_workspaces(context.cst_workspaces)
    context.bin_workspace_byte_arrays = __gen_workspaces(context.bin_workspaces)
    del context.original_input_arrays
    # Following actions need to acquire global lock
    __notify_status("OnAcquireLock")
    device_id = [dev_id]
    with subprocess_device_locks[dev_id] if not context.model else nullcontext():
        context.device_id = device_id[0]
        if not context.model:
            get_process_context().get_lock(subprocess_device_locks[dev_id])
        __notify_status("OnProfilingPrint")
        __profiling_print(context)
        __notify_status("OnDynProfiling")
        context.dyn_prof_result = do_profiling(context, "dynamic")
        __notify_status("OnStcProfiling")
        context.stc_prof_result = do_profiling(context, "static")
        __notify_status("OnCstProfiling")
        context.cst_prof_result = do_profiling(context, "const")
        __notify_status("OnBinProfiling")
        context.bin_prof_result = do_profiling(context, "binary")
    if not context.model:
        get_process_context().release_lock(subprocess_device_locks[dev_id])
    __notify_status("PostProfiling")
    passed, cst_passed = handle_profiling_result(context)
    __notify_status("OnDataTransformation")
    __data_postprocessing(context)
    __notify_status("OnDumpDataIfRequired")
    __dump_data(context)
    del context.dyn_input_byte_arrays
    del context.stc_input_byte_arrays
    del context.output_byte_arrays
    del context.dyn_tiling_data_bytes
    __notify_status("ExtractOutputForComparison")
    bin_outputs_wo_workspace, cst_outputs_wo_workspace, dyn_outputs_wo_workspace, stc_outputs_wo_workspace = \
        __extract_output(context)
    del context.dyn_workspace_byte_arrays
    del context.stc_workspace_byte_arrays
    del context.cst_workspace_byte_arrays
    del context.bin_workspace_byte_arrays
    __notify_status("OnComparison")
    compare_result = comparing(context.dyn_kernel_name, context.stc_kernel_name,
                               context.cst_kernel_name, context.bin_kernel_name,
                               dyn_outputs_wo_workspace,
                               stc_outputs_wo_workspace,
                               cst_outputs_wo_workspace,
                               bin_outputs_wo_workspace,
                               context.golden_arrays,
                               context.precision_tolerances,
                               context.output_dtypes,
                               context.strict_precision_mode,
                               context.absolute_precision,
                               max(math.ceil(max([shape_product(output) for output in context.stc_outputs
                                                  if output is not None]) * get_dtype_width(
                                   context.output_dtypes[0])
                                             / (1024 * 1024 * 256)), 1))
    __notify_status("OnReturning")
    return_structure = ProfilingReturnStructure()
    return_structure.construct(context, compare_result, passed, cst_passed, input_memory_size, output_memory_size)
    __profiling_end_print(context, compare_result, passed)
    del context.dyn_prof_result.output_bytes
    del context.stc_prof_result.output_bytes
    del context.cst_prof_result.output_bytes
    del context.bin_prof_result.output_bytes
    del context
    del dyn_outputs_wo_workspace, stc_outputs_wo_workspace, cst_outputs_wo_workspace, bin_outputs_wo_workspace
    gc.collect()
    return return_structure


def __waiting_for_memory():
    print_once = False
    while psutil.virtual_memory().available <= psutil.virtual_memory().total * 0.5:
        if not print_once:
            logging.warning("Task paused because of insufficient memory!")
            print_once = True
        time.sleep(1)


def __extract_output(context):
    dyn_output_size = max(1, len(context.dyn_prof_result.output_bytes) - len(context.dyn_workspace_byte_arrays))
    stc_output_size = max(1, len(context.stc_prof_result.output_bytes) - len(context.stc_workspace_byte_arrays))
    cst_output_size = max(1, len(context.cst_prof_result.output_bytes) - len(context.cst_workspace_byte_arrays))
    bin_output_size = max(1, len(context.bin_prof_result.output_bytes) - len(context.bin_workspace_byte_arrays))
    dyn_outputs_wo_workspace = context.dyn_prof_result.output_bytes[:dyn_output_size]
    stc_outputs_wo_workspace = context.stc_prof_result.output_bytes[:stc_output_size]
    cst_outputs_wo_workspace = context.cst_prof_result.output_bytes[:cst_output_size]
    bin_outputs_wo_workspace = context.bin_prof_result.output_bytes[:bin_output_size]
    return bin_outputs_wo_workspace, cst_outputs_wo_workspace, dyn_outputs_wo_workspace, stc_outputs_wo_workspace


def __profiling_end_print(context, compare_result, passed):
    logging.info("\n########################\n"
                 "Performance result: \n"
                 "DYN_PERF: %s\n"
                 "STC_PERF: %s\n"
                 "CST_PERF: %s\n"
                 "BIN_PERF: %s\n"
                 "STATUS: %s\n"
                 "########################\n"
                 "Comparison result: \n"
                 "DYN_GOLD: %s\n"
                 "STC_GOLD: %s\n"
                 "RELATIVE: %s\n"
                 "CST_GOLD: %s\n"
                 "CST_RELATIVE: %s\n"
                 "BIN_GOLD: %s\n"
                 "BIN_RELATIVE: %s\n"
                 "STATUS: %s\n"
                 "########################\n" % (context.dyn_prof_result.cycle,
                                                 context.stc_prof_result.cycle,
                                                 context.cst_prof_result.cycle,
                                                 context.bin_prof_result.cycle,
                                                 passed,
                                                 compare_result[0],
                                                 compare_result[1],
                                                 compare_result[2],
                                                 compare_result[3],
                                                 compare_result[4],
                                                 compare_result[5],
                                                 compare_result[6],
                                                 compare_result[7]))


def __parse_binary_tiling_data(context):
    # noinspection PyBroadException
    try:
        context.bin_tiling_data_bytes, context.bin_tuple_tiling_data = parse_tiling_data(context.bin_tiling_data)
        context.bin_str_tiling_data, context.bin_is_tik = \
            get_str_tiling_data(context.bin_tuple_tiling_data, context.bin_compile_info, context.bin_tiling_key)
    except:
        logging.exception("Binary tiling data parsing failure")
        context.bin_compile_result = "TILING_PARSE_FAILURE"


def __parse_dynamic_tiling_data(context):
    # noinspection PyBroadException
    try:
        context.dyn_tiling_data_bytes, context.dyn_tuple_tiling_data = parse_tiling_data(context.dyn_tiling_data)
        context.dyn_str_tiling_data, context.is_tik = \
            get_str_tiling_data(context.dyn_tuple_tiling_data, context.dyn_compile_info, context.dyn_tiling_key)
    except:
        logging.exception("Dynamic tiling data parsing failure")
        context.dyn_compile_result = "TILING_PARSE_FAILURE"


def __parse_manual_params(context):
    # noinspection PyBroadException
    try:
        dyn_block_dim = context.dyn_block_dim
        stc_block_dim = context.stc_block_dim
        cst_block_dim = context.cst_block_dim
        bin_block_dim = context.bin_block_dim
        if context.manual_block_dim:
            try:
                if get(context.manual_block_dim, 0):
                    dyn_block_dim = get(context.manual_block_dim, 0)
                if get(context.manual_block_dim, 1):
                    stc_block_dim = get(context.manual_block_dim, 1)
                if get(context.manual_block_dim, 2):
                    cst_block_dim = get(context.manual_block_dim, 2)
                if get(context.manual_block_dim, 3):
                    bin_block_dim = get(context.manual_block_dim, 3)
            except:
                raise RuntimeError(f"manual_block_dim should be a tuple, not {context.manual_block_dim}") from None
    except:
        logging.exception("Manual block dim parsing failure")
    else:
        context.dyn_block_dim = dyn_block_dim
        context.stc_block_dim = stc_block_dim
        context.cst_block_dim = cst_block_dim
        context.bin_block_dim = bin_block_dim
    if context.manual_dyn_workspaces:
        context.dyn_workspaces = context.manual_dyn_workspaces
    if context.manual_stc_workspaces:
        context.stc_workspaces = context.manual_stc_workspaces
    # noinspection PyBroadException
    try:
        dyn_tiling_data = context.dyn_tiling_data
        dyn_tiling_key = context.dyn_tiling_key
        # Check manual data and move self dyn_tiling_key to here
        if context.manual_tiling_data:
            dyn_tiling_data = context.manual_tiling_data[1:]
            dyn_tiling_key = context.manual_tiling_data[0]
    except:
        logging.exception("Manual tiling data parsing failure")
    else:
        context.dyn_tiling_data = dyn_tiling_data
        context.dyn_tiling_key = dyn_tiling_key


def __profiling_print(context):
    logging.info(
        "\n====================================================================\n" +
        "=======================================================\n" +
        "==================================\n" +
        "Profiling unit received parameters: \n" +
        "op_type: %s\n" % str(context.op_name) +
        "DynShape Input Shape: %s\n" % str(context.dyn_inputs) +
        "DynShape Kernel Name: %s\n" % str(context.dyn_kernel_name) +
        "DynShape Compilation Status: %s\n" % str(context.dyn_compile_result) +
        "%s\n" % str(get_global_storage().dyn_switches) +
        "DynShape Input Data Bytes: %s\n" % str(tuple(len(iput) for iput in context.dyn_input_byte_arrays)) +
        "DynShape BlockDim: %s\n" % str(context.dyn_block_dim) +
        "DynShape Workspace Bytes: %s\n" % str(context.dyn_workspaces) +
        "DynShape Tiling Data Parsed Dict: %s\n" % context.dyn_str_tiling_data +
        "DynShape Tiling Data Parsed Tuple: %s\n" % str(context.dyn_tuple_tiling_data) +
        "DynShape Tiling Data RAW: %s\n" % str(context.dyn_tiling_data_bytes) +
        "DynShape Tiling Key: %s\n" % str(context.dyn_tiling_key) +
        "////////////////////////////\n" +
        "StcShape Input Shape: %s\n" % str(context.stc_inputs) +
        "StcShape Kernel Name: %s\n" % str(context.stc_kernel_name) +
        "StcShape Compilation Status: %s\n" % str(context.stc_compile_result) +
        "%s\n" % str(get_global_storage().stc_switches) +
        "StcShape Input Data Bytes: %s\n" % str(tuple(len(iput) for iput in context.stc_input_byte_arrays)) +
        "StcShape BlockDim: %s\n" % str(context.stc_block_dim) +
        "StcShape Workspace Bytes: %s\n" % str(context.stc_workspaces) +
        "////////////////////////////\n" +
        "CstShape Kernel Name: %s\n" % str(context.cst_kernel_name) +
        "CstShape Compilation Status: %s\n" % str(context.cst_compile_result) +
        "%s\n" % str(get_global_storage().cst_switches) +
        "CstShape Input Data Bytes: %s\n" % str(tuple(len(iput) for iput in context.dyn_input_byte_arrays)) +
        "CstShape BlockDim: %s\n" % str(context.cst_block_dim) +
        "CstShape Workspace Bytes: %s\n" % str(context.cst_workspaces) +
        "////////////////////////////\n" +
        "Binary Release Kernel Name: %s\n" % str(context.bin_kernel_name) +
        "Binary Release Compilation Status: %s\n" % str(context.bin_compile_result) +
        "%s\n" % str(get_global_storage().bin_switches) +
        "Binary Release BlockDim: %s\n" % str(context.bin_block_dim) +
        "Binary Release Workspace Bytes: %s\n" % str(context.bin_workspaces) +
        "Binary Release Tiling Data Parsed Dict: %s\n" % context.bin_str_tiling_data +
        "Binary Release Tiling Data Parsed Tuple: %s\n" % str(context.bin_tuple_tiling_data) +
        "Binary Release Tiling Data RAW: %s\n" % str(context.bin_tiling_data_bytes) +
        "Binary Release Tiling Key: %s\n" % str(context.bin_tiling_key) +
        "////////////////////////////\n" +
        "Input data dtype: %s\n" % str(context.dyn_input_dtypes) +
        "Output shape: %s\n" % str(context.stc_outputs) +
        "Output data size: %s Byte\n" % str(tuple("inplace%d" % output if isinstance(output, int) else
                                                  len(output) for output in context.output_byte_arrays)) +
        "Output data dtype: %s\n" % str(context.output_dtypes) +
        "Input data range: %s\n" % str(context.actual_input_data_ranges) +
        "Precision Tolerance: %s\n" % str(context.precision_tolerances) +

        "Model Mode: %s\n" % str(context.model) +
        "PID: %d\n" % os.getpid() +
        "Device: %d \n" % context.device_id +
        "==================================\n"
        "=======================================================\n" +
        "===================================================================="
    )


def __dump_data(context: UniversalTestcaseStructure) -> NoReturn:
    # Input
    dump_input_name = context.testcase_name if context.dump_input_data_name is None else context.dump_input_data_name
    dump_output_name = context.testcase_name if context.dump_output_data_name is None else context.dump_output_data_name
    if get_global_storage().dump_mode.value & 0b100 == 0b100:
        for idx, _input in enumerate(context.dyn_input_byte_arrays):
            if isinstance(_input, bytes):
                with open(dump_input_name + "_dyn_input_%d.bin" % idx, "wb+") as f:
                    f.write(_input)
        for idx, _input in enumerate(context.stc_input_byte_arrays):
            if isinstance(_input, bytes):
                with open(dump_input_name + "_stc_input_%d.bin" % idx, "wb+") as f:
                    f.write(_input)
        if isinstance(context.dyn_tiling_data_bytes, bytes):
            with open(dump_input_name + "_dyn_tiling_data.bin", "wb+") as f:
                f.write(context.dyn_tiling_data_bytes)
        if isinstance(context.bin_tiling_data_bytes, bytes):
            with open(dump_input_name + "_bin_tiling_data.bin", "wb+") as f:
                f.write(context.bin_tiling_data_bytes)
    # Output
    if get_global_storage().dump_mode.value & 0b010 == 0b010:
        for idx, _input in enumerate(context.dyn_prof_result.output_bytes):
            if not isinstance(_input, str):
                with open(dump_output_name + "_dyn_output_%d.bin" % idx, "wb+") as f:
                    f.write(_input)
        for idx, _input in enumerate(context.stc_prof_result.output_bytes):
            if not isinstance(_input, str):
                with open(dump_output_name + "_stc_output_%d.bin" % idx, "wb+") as f:
                    f.write(_input)
        for idx, _input in enumerate(context.cst_prof_result.output_bytes):
            if not isinstance(_input, str):
                with open(dump_output_name + "_cst_output_%d.bin" % idx, "wb+") as f:
                    f.write(_input)
        for idx, _input in enumerate(context.bin_prof_result.output_bytes):
            if not isinstance(_input, str):
                with open(dump_output_name + "_bin_output_%d.bin" % idx, "wb+") as f:
                    f.write(_input)
    # Golden
    if get_global_storage().dump_mode.value & 0b001 == 0b001:
        for idx, _input in enumerate(context.golden_arrays):
            if not isinstance(_input, str):
                with open(dump_output_name + "_golden_%d.bin" % idx, "wb+") as f:
                    f.write(_input)


def __data_postprocessing(context: UniversalTestcaseStructure):
    from ...user_defined_modules.postprocessing_funcs.registry import postprocessing_func
    if context.op_name in postprocessing_func:
        (context.dyn_prof_result.output_bytes, context.stc_prof_result.output_bytes,
         context.cst_prof_result.output_bytes, context.bin_prof_result.output_bytes) = \
            postprocessing_func[context.op_name](context)


def do_profiling(context: UniversalTestcaseStructure, mode: str) -> RTSProfilingResult:
    """
    RTS Profiling wrapper
    :param context:
    :param mode:
    :return:
    """

    class _RTSProfilingParam:
        def __init__(self,
                     compile_result: str,
                     tiling_key: int,
                     kernel_name: str,
                     block_dim: int,
                     input_byte_array: tuple,
                     tiling_data_bytes: bytes,
                     output_with_workspaces: tuple,
                     switch: bool,
                     model: bool,
                     device_id: int,
                     use_static_kernel: bool,
                     rts_online_prof: bool,
                     is_valid: bool,
                     fail_reason: str):
            self.compile_result = compile_result
            self.tiling_key = tiling_key
            self.kernel_name = kernel_name
            self.block_dim = block_dim
            self.input_byte_array = input_byte_array
            self.tiling_data_bytes = tiling_data_bytes
            self.output_with_workspaces = output_with_workspaces
            self.switch = switch
            self.model = model
            self.device_id = device_id
            self.use_static_kernel = use_static_kernel
            self.rts_online_prof = rts_online_prof
            self.is_valid = is_valid
            self.fail_reason = fail_reason

    result = RTSProfilingResult()
    param_map = {
        "dynamic": (context.dyn_compile_result,  # 0
                    context.dyn_tiling_key,  # 1
                    context.dyn_kernel_name,  # 2
                    context.dyn_block_dim,  # 3
                    context.dyn_input_byte_arrays,  # 4
                    context.dyn_tiling_data_bytes,  # 5
                    context.output_byte_arrays + context.dyn_workspace_byte_arrays,  # 6
                    get_global_storage().dyn_switches.get_prof(),  # 7
                    context.model,  # 8
                    context.device_id,  # 9
                    context.is_tik,  # 10
                    get_global_storage().dyn_switches.rts_prof,  # 11
                    context.dyn_is_valid,  # 12
                    context.dyn_fail_reason),  # 13
        "static": (context.stc_compile_result,  # 0
                   None,  # 1
                   context.stc_kernel_name,  # 2
                   context.stc_block_dim,  # 3
                   context.stc_input_byte_arrays,  # 4
                   None,  # 5
                   context.output_byte_arrays + context.stc_workspace_byte_arrays,  # 6
                   get_global_storage().stc_switches.get_prof(),  # 7
                   context.model,  # 8
                   context.device_id,  # 9
                   True,  # 10
                   get_global_storage().stc_switches.rts_prof,  # 11
                   context.is_valid,  # 12
                   context.fail_reason),  # 13
        "const": (context.cst_compile_result,  # 0
                  None,  # 1
                  context.cst_kernel_name,  # 2
                  context.cst_block_dim,  # 3
                  context.dyn_input_byte_arrays,  # 4
                  None,  # 5
                  context.output_byte_arrays + context.cst_workspace_byte_arrays,  # 6
                  get_global_storage().cst_switches.get_prof(),  # 7
                  context.model,  # 8
                  context.device_id,  # 9
                  True,  # 10
                  get_global_storage().cst_switches.rts_prof,  # 11
                  context.dyn_is_valid,  # 12
                  context.dyn_fail_reason),  # 13
        "binary": (context.bin_compile_result,  # 0
                   context.bin_tiling_key,  # 1
                   context.bin_kernel_name,  # 2
                   context.bin_block_dim,  # 3
                   context.dyn_input_byte_arrays,  # 4
                   context.bin_tiling_data_bytes,  # 5
                   context.output_byte_arrays + context.bin_workspace_byte_arrays,  # 6
                   get_global_storage().bin_switches.get_prof(),  # 7
                   context.model,  # 8
                   context.device_id,  # 9
                   context.bin_is_tik,  # 10
                   get_global_storage().bin_switches.rts_prof,  # 11
                   context.dyn_is_valid,  # 12
                   context.dyn_fail_reason),  # 13
    }
    logging.debug("Entering RTS profiling sequence with %s mode" % mode)
    param = _RTSProfilingParam(*param_map[mode])
    if param.switch:
        if not param.compile_result == "SUCC":
            result.cycle = param.compile_result
            result.output_bytes = (param.compile_result,)
            result.pmu = (param.compile_result,) * get_global_storage().PMU_RETURN_SIZE
        elif not param.is_valid:
            result.cycle = param.fail_reason
            result.output_bytes = (param.fail_reason,)
            result.pmu = (param.fail_reason,) * get_global_storage().PMU_RETURN_SIZE
        elif not param.block_dim > 0:
            result.cycle = "INVALID_TILING"
            result.output_bytes = ("INVALID_TILING",)
            result.pmu = ("INVALID_TILING",) * get_global_storage().PMU_RETURN_SIZE
        else:
            if param.model:
                # Get all path and names
                kernel_meta = get_global_storage().kernel_meta
                root_dir = tbetoolkits.__path__[0]
                model_path = pathlib.Path(root_dir, "../model_results/%s/%s/" % (context.testcase_name, mode))
                models_dir = pathlib.Path(root_dir, "../models")
                arch_name = plat.uname()[-1]
                config_file_name = "%s_%s_%s_config.zip" % (arch_name,
                                                            get_global_storage().mode.is_model(),
                                                            get_global_storage().device_platform)
                config_file_path = pathlib.Path(models_dir, "%s" % config_file_name)
                # IO
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                os.makedirs(pathlib.Path(model_path, kernel_meta), exist_ok=True)
                if get_global_storage().mode == tbetoolkits.utilities.MODE.ASCEND_ESLMODEL:
                    os.makedirs(pathlib.Path(model_path, "log"), exist_ok=True)
                shutil.copy2(pathlib.Path(kernel_meta, "%s.o" % param.kernel_name),
                             pathlib.Path(model_path, kernel_meta, "%s.o" % param.kernel_name))
                if get_global_storage().model_update_configs:
                    os.system("unzip -o -qq %s -d %s" % (config_file_path, model_path))
                os.chdir(model_path)
            device = None
            # noinspection PyBroadException
            try:
                if "device" in get_process_context().storage:
                    device = get_process_context().storage["device"]
                else:
                    device = RTSInterface(param.model)
                    device.set_device(param.device_id)
                    get_process_context().storage["device"] = device
                if device.device_id is None:
                    device = RTSInterface(param.model)
                    device.set_device(param.device_id)
                    get_process_context().storage["device"] = device
                if param.use_static_kernel:
                    run_kernel_name = param.kernel_name + "__kernel0"
                else:
                    run_kernel_name = param.kernel_name + "_" + str(numpy.uint32(param.tiling_key))
                result.cycle, result.output_bytes, result.pmu = \
                    rtsProfilingCallEx(device, int(param.block_dim),
                                       run_kernel_name,
                                       param.kernel_name,
                                       param.input_byte_array,
                                       param.output_with_workspaces,
                                       param.tiling_data_bytes,
                                       param.model, get_global_storage().run_time,
                                       param.tiling_key, param.rts_online_prof)
            except:
                root_dir = tbetoolkits.__path__[0]
                os.chdir(pathlib.Path(root_dir, "../"))
                logging.exception("Profiling Sequence of mode %s failed:" % mode)
                result.cycle = "PROFILING_FAILURE"
                result.output_bytes = ("PROFILING_FAILURE",)
                result.pmu = ("UNKNOWN",) * get_global_storage().PMU_RETURN_SIZE
                if not param.model:
                    os.system(f"mkdir -p errors/{os.getpid()} && cd errors/{os.getpid()} && msnpureport && cd -")
                if device:
                    device.reset()
            finally:
                if param.model:
                    os.chdir("./../../../")
    else:
        result.cycle = "SUPPRESSED"
        result.output_bytes = ("SUPPRESSED",)
        result.pmu = ("UNKNOWN",) * get_global_storage().PMU_RETURN_SIZE
    return result


def handle_profiling_result(context: UniversalTestcaseStructure):
    """
    Returns parsed cycle counts and passing state
    :param context:
    :return:
    """
    # noinspection PyBroadException
    def _get_cycle(cycle, off_flag):
        if str(cycle) == off_flag:
            return "PASS", off_flag

        _passed, _cycle_f = "EXCEPTION", "RTS_PROF_INVALID"
        try:
            _cycle_f = float(cycle)
            if _cycle_f <= 0:
                _cycle_f = "RTS_PROF_INVALID"
            else:
                _passed = "PASS"
        except:
            _cycle_f = "RTS_PROF_INVALID"
        finally:
            return _passed, _cycle_f

    passed, cst_passed = "EXCEPTION", "EXCEPTION"

    _, stc_cycle_f = _get_cycle(context.stc_prof_result.cycle, "STC_OFF")
    dyn_pass, dyn_cycle_f = _get_cycle(context.dyn_prof_result.cycle, "DYN_OFF")
    cst_pass, cst_cycle_f = _get_cycle(context.cst_prof_result.cycle, "CST_OFF")

    if isinstance(stc_cycle_f, str):
        passed = dyn_pass if stc_cycle_f == "STC_OFF" else "EXCEPTION"
        cst_passed = cst_pass if stc_cycle_f == "STC_OFF" else "EXCEPTION"
    else:
        if isinstance(dyn_cycle_f, str):
            passed = "PASS" if dyn_cycle_f == "DYN_OFF" else "EXCEPTION"
        else:
            passed = "PASS" if (stc_cycle_f / dyn_cycle_f >= get_global_storage().perf_threshold[0] or
                                dyn_cycle_f - stc_cycle_f <= get_global_storage().perf_threshold[1]) \
                else "FAIL"
            context.dyn_prof_result.cycle = dyn_cycle_f
            context.stc_prof_result.cycle = stc_cycle_f

        if isinstance(cst_cycle_f, str):
            cst_passed = "PASS" if cst_cycle_f == "CST_OFF" else "EXCEPTION"
        else:
            cst_passed = "PASS" if (stc_cycle_f / cst_cycle_f >= get_global_storage().perf_threshold[0] or
                                cst_cycle_f - stc_cycle_f <= get_global_storage().perf_threshold[1]) \
                else "FAIL"
            context.cst_prof_result.cycle = cst_cycle_f

    return passed, cst_passed
