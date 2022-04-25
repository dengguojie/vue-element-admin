#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
RTS Profiling Sequence for Universal testcases
"""
# Standard Packages
import json
import ctypes
import logging
import pathlib
from typing import Optional, Tuple, Union

# Third-party Packages
import numpy
import tbetoolkits
from ..driver import DRVInterface
from ..runtime import RTSInterface
from ..model2trace import parse_dumps_in_folder
from ...utilities import get_global_storage
from ...utilities import get_process_name


def rtsProfilingCallEx(device: RTSInterface, block_dim: int, run_kernel_name: str, kernel_name: str,
                       input_byte_arrays: tuple, output_byte_arrays: tuple, arg_byte_array: Optional[bytes],
                       model, repeat_time, possible_tiling_key,
                       rts_online_prof):
    """
    Extended rts profiling call
    """
    if model:
        return _rtsModelingCall(device, block_dim, run_kernel_name, kernel_name, input_byte_arrays, output_byte_arrays,
                                arg_byte_array, possible_tiling_key)
    else:
        return _rtsProfilingCall(device, block_dim, run_kernel_name, kernel_name,
                                 input_byte_arrays, output_byte_arrays, arg_byte_array,
                                 repeat_time, possible_tiling_key, rts_online_prof)


def _rtsProfilingCall(device: RTSInterface, block_dim: int, run_kernel_name, kernel_name: str,
                      input_byte_arrays: Tuple[bytes],
                      output_byte_arrays: Tuple[Union[bytes, int]], arg_byte_array: Optional[bytes],
                      repeat_time: int, possible_tiling_key,
                      rts_online_prof: bool = True):
    # Check
    if repeat_time < 0:
        raise ValueError("Repeat time has to be higher than 0")
    ################################
    # Establish RTS Context for device
    ################################
    device.create_context("RT_CTX_NORMAL_MODE")
    pmu_status = get_global_storage().PMU
    ################################
    # Kernel Registration
    ################################
    kernel_meta = get_global_storage().kernel_meta
    try:
        registered_binary = device.register_device_binary_kernel(str(pathlib.Path(kernel_meta, "%s.o" % kernel_name)),
                                                                 get_global_storage().core_type)
    except RuntimeError:
        logging.exception(f"RTS Register Binary failed, kernel object {kernel_name}.o does not exist or is invalid.")
        device.destroy_context()
        total_cycle = "RTS_BINARY_FAILURE"
        actual_result_byte_arrays = ("RTS_BINARY_FAILURE",) * len(output_byte_arrays)
        pmu_result = ["RTS_BINARY_FAILURE"] * get_global_storage().PMU_RETURN_SIZE
        return total_cycle, actual_result_byte_arrays, pmu_result
    # noinspection PyBroadException
    try:
        stubfunc_p = device.register_function(registered_binary,
                                              "%s" % run_kernel_name,
                                              0)
    except:
        old_run_kernel_name = run_kernel_name
        if run_kernel_name.endswith("__kernel0") and possible_tiling_key is not None:
            run_kernel_name = "__".join(run_kernel_name.split("__")[:-1]) \
                              + "_" + str(numpy.uint32(possible_tiling_key))
        else:
            run_kernel_name = "_".join(run_kernel_name.split("_")[:-1]) + "__kernel0"
        # noinspection PyBroadException
        try:
            stubfunc_p = device.register_function(registered_binary,
                                                  "%s" % run_kernel_name,
                                                  0)
        except:
            logging.error(f"RTS Register Function failed, expect symbol {old_run_kernel_name} or {run_kernel_name} "
                          f"in {kernel_name}.o")
            device.destroy_context()
            total_cycle = "RTS_FUNCTION_FAILURE"
            actual_result_byte_arrays = ("RTS_FUNCTION_FAILURE",) * len(output_byte_arrays)
            pmu_result = ["RTS_FUNCTION_FAILURE"] * get_global_storage().PMU_RETURN_SIZE
            return total_cycle, actual_result_byte_arrays, pmu_result
    ################################
    # Pre-Kernel Checks
    ################################
    if get_global_storage().PMU and get_global_storage().device_platform in ["Ascend710"]:
        logging.warning("Online profiling has been disabled because PMU conflicts with it on %s"
                        % get_global_storage().device_platform)
        rts_online_prof = False
    status, actual_result_byte_arrays, pmu_result, profiling_data = __rts_kernel_sequence(device, stubfunc_p,
                                                                                          input_byte_arrays,
                                                                                          output_byte_arrays,
                                                                                          arg_byte_array,
                                                                                          block_dim,
                                                                                          1,
                                                                                          with_output=False,
                                                                                          rts_online_prof=False,
                                                                                          pmu_status=False)
    if status != "OK":
        profiling_data = [status]
        pmu_result = ["UNKNOWN"] * get_global_storage().PMU_RETURN_SIZE
    else:
        status, actual_result_byte_arrays, pmu_result, profiling_data = \
            __rts_kernel_sequence(device, stubfunc_p,
                                  input_byte_arrays, output_byte_arrays,
                                  arg_byte_array,
                                  block_dim,
                                  repeat_time,
                                  with_output=True,
                                  rts_online_prof=rts_online_prof,
                                  pmu_status=pmu_status)
    # noinspection PyBroadException
    try:
        device.unregister_device_binary_kernel(registered_binary)
    except:
        logging.exception("Unregister device binary kernel failed:")
    device.destroy_context()
    # Check if profiling result is valid
    profiling_data_valid = all([isinstance(profiling_data_unit, int) for profiling_data_unit in profiling_data])
    logging.debug(f"RTS Online Profiling result: {profiling_data}")
    if profiling_data_valid:
        profiling_data.sort()
        total_cycle = numpy.median(profiling_data)
    else:
        profiling_data = tuple(map(str, profiling_data))
        total_cycle = ", ".join(profiling_data)
    return total_cycle, actual_result_byte_arrays, tuple(pmu_result)


def __rts_kernel_sequence(device, stubfunc_p,
                          input_byte_arrays, output_byte_arrays, arg_byte_array,
                          block_dim, repeat_time, with_output=None, rts_online_prof=None, pmu_status=None):
    def __free_memories(memory_addrs):
        for addr in memory_addrs:
            device.free(addr)

    drv = DRVInterface()
    ################################
    # Profiling Preparation
    ################################
    if with_output:
        actual_result_byte_arrays = []
    else:
        actual_result_byte_arrays = ["NO_OUTPUT"] * len(output_byte_arrays)
    pmu_result = ["UNKNOWN"] * get_global_storage().PMU_RETURN_SIZE
    profiling_data = []
    status = "OK"
    ################################
    # Profiling Main Sequence
    ################################
    for repeat_idx in range(repeat_time):
        ################################
        # Prepare Memory on HBM
        ################################
        arg_mem_addrs, input_mem_addrs, output_mem_addrs = __rts_prepare_hbm(device, input_byte_arrays,
                                                                             output_byte_arrays, arg_byte_array)
        ################################
        # Launch Kernel
        ################################
        if rts_online_prof:
            device.start_online_profiling(None, 1)
        with drv.PMU(device,
                     get_global_storage().PMU_MODE,
                     pmu_result,
                     pmu_status and repeat_idx == repeat_time - 1):
            # noinspection PyBroadException
            try:
                device.launch_kernel(stubfunc_p,
                                     block_dim,
                                     (*input_mem_addrs,
                                      *output_mem_addrs,
                                      *arg_mem_addrs),
                                     len(input_mem_addrs) + len(output_mem_addrs) + len(arg_mem_addrs),
                                     None, None)
            except:
                logging.exception("RTSProfilingCall encountered an unknown rts error during kernel launch stage:")
                status = "LAUNCH_FAILED"
            else:
                try:
                    device.synchronize_with_stream(None)
                except RuntimeError as e:
                    if "AICORE_TRAP_EXCEPTION" in e.args[0]:
                        status = "TRAP"
                        logging.error("Reached AICORE Trap Exception")
                    elif "AICORE_EXCEPTION" in e.args[0]:
                        status = "AIC_ERROR"
                        logging.error(f"AIC_ERROR encountered with rts profiling status {rts_online_prof}")
                    elif "AICORE_TIMEOUT" in e.args[0]:
                        status = "TIMEOUT"
                        logging.error("AIC Task TIMEOUT")
                    else:
                        status = "UNKNOWN_RTS_ERROR"
                        logging.exception("RTSProfilingCall encountered an unknown rts error during finish stage:")
            if rts_online_prof:
                original_prof_data = device.get_online_profiling_data(None, 1)
                device.stop_online_profiling(None)
                raw_prof_data = original_prof_data[0]
                profiling_data.append(raw_prof_data.totalcycle)
            else:
                profiling_data.append(status)

        ################################
        # Collect Data
        ################################
        if repeat_idx == repeat_time - 1 and with_output:
            for idx, output_byte_array in enumerate(output_byte_arrays):
                if not isinstance(output_byte_array, int):
                    actual_result_byte_arrays.append(
                        device.get_data_from_hbm(output_mem_addrs[idx],
                                                 len(output_byte_array)))
                else:
                    actual_result_byte_arrays.append(device.get_data_from_hbm(
                        input_mem_addrs[output_byte_array],
                        len(input_byte_arrays[output_byte_array])))
        ################################
        # Free memory
        ################################
        __free_memories(input_mem_addrs)
        __free_memories(arg_mem_addrs)
        free_needed_addrs = []
        for idx, output_byte_array in enumerate(output_byte_arrays):
            if not isinstance(output_byte_array, int):
                free_needed_addrs.append(output_mem_addrs[idx])
        __free_memories(free_needed_addrs)
        if status != "OK":
            break
    return status, actual_result_byte_arrays, pmu_result, profiling_data


def __rts_prepare_hbm(device, input_byte_arrays, output_byte_arrays, arg_byte_array):
    input_mem_addrs = []
    for input_idx, input_byte_array in enumerate(input_byte_arrays):
        input_mem_addrs.append(device.copy_bin_to_hbm(input_byte_array))
    arg_mem_addrs = []
    if arg_byte_array is not None:
        arg_mem_addrs.append(device.copy_bin_to_hbm(arg_byte_array))
    output_mem_addrs = []
    for output_idx, output_byte_array in enumerate(output_byte_arrays):
        if not isinstance(output_byte_array, int):
            output_mem_addrs.append(device.copy_bin_to_hbm(output_byte_array))
        else:
            output_mem_addrs.append(input_mem_addrs[output_byte_array])
    return arg_mem_addrs, input_mem_addrs, output_mem_addrs


def _rtsModelingCall(device: RTSInterface, block_dim: int, run_kernel_name: str, kernel_name: str,
                     input_byte_arrays: tuple, output_byte_arrays: tuple, arg_byte_array: Optional[bytes],
                     possible_tiling_key: int = None):
    def __free_memories(memory_addrs):
        for addr in memory_addrs:
            device.free(ctypes.c_void_p(addr))

    ################################
    # Establish RTS Context for device
    ################################
    device.create_context("RT_CTX_NORMAL_MODE")
    ################################
    # Kernel Registration
    ################################
    kernel_meta = get_global_storage().kernel_meta
    registered_binary = device.register_device_binary_kernel(str(pathlib.Path(kernel_meta, "%s.o" % kernel_name)),
                                                             get_global_storage().core_type)
    # noinspection PyBroadException
    try:
        stubfunc_p = device.register_function(registered_binary,
                                              "%s" % run_kernel_name,
                                              0)
    except:
        logging.error("RTS Register Function failed, expect symbol %s in %s.o" % (run_kernel_name, kernel_name))
        if run_kernel_name.endswith("__kernel0"):
            run_kernel_name = "__".join(run_kernel_name.split("__")[:-1]) \
                              + "_" + str(numpy.uint32(possible_tiling_key))
        else:
            run_kernel_name = "_".join(run_kernel_name.split("_")[:-1]) + "__kernel0"
        # noinspection PyBroadException
        try:
            stubfunc_p = device.register_function(registered_binary,
                                                  "%s" % run_kernel_name,
                                                  0)
        except:
            logging.exception("RTS Register Function failed, expect symbol %s in %s.o" % (run_kernel_name, kernel_name))
            raise
    ################################
    # Profiling Cycles Preparation
    ################################
    actual_result_byte_arrays = []
    total_cycle = ""
    ################################
    # Prepare Memory on HBM
    ################################
    input_mem_addrs = []
    for input_byte_array in input_byte_arrays:
        input_mem_addrs.append(device.copy_bin_to_hbm(input_byte_array).value)
    arg_mem_addrs = []
    if arg_byte_array is not None:
        arg_mem_addrs.append(device.copy_bin_to_hbm(arg_byte_array).value)
    output_mem_addrs = []
    for output_byte_array in output_byte_arrays:
        if not isinstance(output_byte_array, int):
            output_mem_addrs.append(device.copy_bin_to_hbm(output_byte_array).value)
        else:
            output_mem_addrs.append(input_mem_addrs[output_byte_array])
    ################################
    # Launch Kernel
    ################################
    # noinspection PyBroadException
    try:
        device.launch_kernel(stubfunc_p,
                             block_dim,
                             (*input_mem_addrs,
                              *output_mem_addrs,
                              *arg_mem_addrs),
                             len(input_mem_addrs) + len(output_mem_addrs) + len(arg_mem_addrs),
                             None, None)
    except:
        total_cycle += "KERNEL_LAUNCH_ERROR" if not total_cycle else ", KERNEL_LAUNCH_ERROR"
        logging.exception("RTSProfilingCall encountered an unknown rts error during kernel launch stage:")
    else:
        try:
            device.synchronize_with_stream(None)
        except RuntimeError as e:
            if "AICORE_TRAP_EXCEPTION" in e.args[0]:
                total_cycle += "AIC_TRAP" if not total_cycle else ", AIC_TRAP"
            elif "AICORE_EXCEPTION" in e.args[0]:
                total_cycle += "AIC_ERROR" if not total_cycle else ", AIC_ERROR"
            elif "AICORE_TIMEOUT" in e.args[0]:
                total_cycle += "TIMEOUT" if not total_cycle else ", TIMEOUT"
            else:
                total_cycle += "UNK_RTS_ERROR" if not total_cycle else ", UNK_RTS_ERROR"
                logging.exception("RTSProfilingCall encountered an unknown rts error during kernel finish stage:")
    ################################
    # Collect Data if needed
    ################################
    for idx, output_byte_array in enumerate(output_byte_arrays):
        if not isinstance(output_byte_array, int):
            actual_result_byte_arrays.append(
                device.get_data_from_hbm(output_mem_addrs[idx],
                                         len(output_byte_array)))
        else:
            actual_result_byte_arrays.append(device.get_data_from_hbm(
                input_mem_addrs[output_byte_array],
                len(input_byte_arrays[output_byte_array])))
    ################################
    # Free memory
    ################################
    __free_memories(input_mem_addrs)
    __free_memories(arg_mem_addrs)
    free_needed_addrs = []
    for idx, output_byte_array in enumerate(output_byte_arrays):
        if not isinstance(output_byte_array, int):
            free_needed_addrs.append(output_mem_addrs[idx])
    __free_memories(free_needed_addrs)
    ################################
    # Stop and Obtain data
    ################################
    device.unregister_device_binary_kernel(registered_binary)
    device.destroy_context()
    device.reset()
    # noinspection PyBroadException
    try:
        end_cycle, total_cycle = __gen_model_trace(get_process_name(),
                                                   run_kernel_name.split("_")[0], get_global_storage().mode)
    except:
        end_cycle = total_cycle = "UNKNOWN"
        logging.exception("Generate model trace failed:")
    return end_cycle, actual_result_byte_arrays, (total_cycle,) * 8


# noinspection DuplicatedCode
def __gen_model_trace(file_name, run_mode, model_mode):
    total_cycle = 0
    end_cycle = 0
    if model_mode == tbetoolkits.utilities.MODE.ASCEND_CAMODEL:
        dump_path = "."
        for block_dim_index in get_global_storage().model_target_block_dim:
            container = parse_dumps_in_folder(dump_path, core_index=block_dim_index,
                                              platform=get_global_storage().device_platform, model_type=model_mode)
            if container.end_event:
                if container.end_event.data["ts"] > end_cycle:
                    end_cycle = container.end_event.data["ts"]
                total_cycle += container.end_event.data["ts"]
            tbetoolkits.third_party.trace2html.Main(json.dumps(container.get()),
                                                    f"{run_mode}_{file_name}_core{block_dim_index}.html")
    elif model_mode == tbetoolkits.utilities.MODE.ASCEND_PEMMODEL:
        dump_path = "./model"
        container = parse_dumps_in_folder(dump_path, core_index=0,
                                          platform=get_global_storage().device_platform, model_type=model_mode)
        if container.end_event:
            if container.end_event.data["ts"] > end_cycle:
                end_cycle = container.end_event.data["ts"]
            total_cycle += container.end_event.data["ts"]
        tbetoolkits.third_party.trace2html.Main(json.dumps(container.get()),
                                                f"{run_mode}_{file_name}_core0.html")
    elif model_mode == tbetoolkits.utilities.MODE.ASCEND_ESLMODEL:
        dump_path = "."
        for block_dim_index in get_global_storage().model_target_block_dim:
            for real_idx in range(block_dim_index * 3, block_dim_index * 3 + 3):
                container = parse_dumps_in_folder(dump_path, core_index=real_idx,
                                                  platform=get_global_storage().device_platform, model_type=model_mode)
                if container.end_event:
                    if container.end_event.data["ts"] > end_cycle:
                        end_cycle = container.end_event.data["ts"]
                    total_cycle += container.end_event.data["ts"]
                tbetoolkits.third_party.trace2html.Main(json.dumps(container.get()),
                                                        f"{run_mode}_{file_name}_unit{real_idx}.html")
    else:
        raise RuntimeError(f"Unknown model: {model_mode}")
    return str(end_cycle), str(total_cycle)
