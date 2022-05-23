#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Dynamic Shape NPU Profiling Structure
"""
# Standard Packages
from typing import Optional
# Third-party Packages
from ..testcase_manager import UniversalTestcaseStructure
from ...utilities import get_global_storage


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
    def construct(self, context: UniversalTestcaseStructure, compare_result, passed, cst_passed, input_size: int, output_size: int):
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
        self.cst_perf_status = cst_passed
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