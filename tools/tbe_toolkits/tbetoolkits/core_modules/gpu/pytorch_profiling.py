#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Profiling method for Tensorflow
"""
# Standard Packages
import logging
import re

# Third-party Packages
from .input_generation import __gen_input
from ..tbe_multiprocessing.pool import get_process_context
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


def pytorch_profiling(context: UniversalTestcaseStructure, device_id):
    __notify_status("OnLoadPytorch")
    from ...user_defined_modules.davinci_to_torch import dav2torch_registry
    import torch
    # Do parameter mapping
    if context.op_name in dav2torch_registry.dav_op_to_torch_map:
        context = dav2torch_registry.dav_op_to_torch_map[context.op_name](context)
    params = {**context.other_compilation_params, **context.other_runtime_params}
    torch_func = context.torch_func
    testcase_name = context.testcase_name
    if torch_func is None:
        logging.error("\n===========================\n"
                      "Operator %s not found in pytorch\n"
                      "===========================\n" % context.op_name)
        return "UNSUPPORTED", "UNSUPPORTED"
    if "SHAPE_OUT_OF_BOUND" in str(context.original_line):
        return "SHAPE_OUT_OF_BOUND", "SHAPE_OUT_OF_BOUND"
    if "OUTPUT_INFERENCE_FAILED" in str(context.original_line):
        return "OUTPUT_INFERENCE_FAILED", "OUTPUT_INFERENCE_FAILED"
    __notify_status("OnGenInput")
    __gen_input(context)
    for idx, input_array in enumerate(context.input_arrays):
        context.input_arrays[idx] = torch.from_numpy(input_array)
    __notify_status("OnPytorchProfiling")
    device = torch.device("cuda:" + str(device_id))
    if "device" in params:
        params["device"] = device
    for idx, input_array in enumerate(context.input_arrays):
        context.input_arrays[idx] = input_array.to(device)
    # input_as_list
    temp_placeholders = []
    if context.stc_input_as_list_distribution:
        last_num = 0
        for num in context.stc_input_as_list_distribution:
            if num == 0:
                temp_placeholders.append(context.input_arrays[last_num])
                last_num += 1
            else:
                temp_placeholders.append(context.input_arrays[last_num:last_num + num])
                last_num += num
        if last_num < len(context.input_arrays):
            for tensor in context.input_arrays[last_num:]:
                temp_placeholders.append(tensor)
        context.input_arrays = temp_placeholders
    for _ in range(get_global_storage().run_time * 10):
        torch_func(*context.input_arrays, **params)
    with torch.autograd.profiler.profile(use_cuda=True) as p:
        for _ in range(get_global_storage().run_time * 10):
            torch_func(*context.input_arrays, **params)
    torch_perf_result = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    logging.debug("\n" + torch_perf_result)
    cuda_time_pattern = re.compile("Self CUDA time total: (.+)")
    search_result = cuda_time_pattern.search(torch_perf_result)
    p.export_chrome_trace(f"./gpu_json/{testcase_name}.json")
    if search_result:
        total_time = search_result.group(1).strip()
    else:
        raise RuntimeError("Perf result unavailable")
    if total_time.endswith("us"):
        perf = float(total_time[:-2]) / get_global_storage().run_time / 10
    elif total_time.endswith("ms"):
        perf = float(total_time[:-2]) * 1000 / get_global_storage().run_time / 10
    elif total_time.endswith("s"):
        perf = float(total_time[:-2]) * 1000 * 1000 / get_global_storage().run_time / 10
    else:
        raise RuntimeError(f"Unknown profiling time: {total_time}")
    return perf
