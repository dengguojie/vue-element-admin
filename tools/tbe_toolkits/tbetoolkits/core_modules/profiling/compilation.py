#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Operator compilation method for universal csv testcases
"""
# Standard Packages
import os
import platform as plat
import subprocess
import time
import json
import pathlib
import logging
from typing import Hashable, Union, Tuple

# Third-Party Packages
import numpy

import tbetoolkits
from ..testcase_manager import UniversalTestcaseStructure
from ..tbe_multiprocessing.pool import get_process_context
from ...utilities import get
from ...utilities import eliminate_scalar_shapes
from ...utilities import get_stc_json_op_data
from ...utilities import get_global_storage
from ...utilities import set_process_name
from ...utilities import set_thread_name
from ...utilities import DynamicCompilationResult
from ...utilities import StaticCompilationResult
from ...utilities import ConstCompilationResult
from ...utilities import BinaryCompilationResult
from ...utilities import DynamicOpTilingResult
from ...utilities import BinaryOpTilingResult
from ..operator.op_interface import OperatorInterface


def process_kernel_string(value: str) -> str:
    """
    String for kernel_name
    :param value:
    :return:
    """
    if not isinstance(value, str):
        raise TypeError("Testcase name must be a string.")
    if len(value) == 0:
        raise RuntimeError("Testcase name must not be empty.")
    valid_char = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    for idx in range(len(value)):
        if value[idx] not in valid_char:
            logging.warning("%s is not a valid kernel name, all invalid character will be converted to _" % value)
            break
    value_container = [v if v in valid_char else "_" if v != "-" else "neg" for v in value]
    result = "".join(value_container)
    if len(result) > 120:
        hash_code = hash(result)
        result = result[:80] + str(hash_code).replace("-", "neg")
    return result


def compilation_process(testcase: UniversalTestcaseStructure,
                        group_id: int,
                        mode: str):
    """
    Universal Operator Compilation Sequence
    :param mode:
    :param group_id:
    :param testcase:
    :return:
    """

    def __notify_status(status: str):
        set_thread_name(status)
        get_process_context().send_data("stage", status)

    def __report_process_name(my_name_is: str):
        set_process_name(my_name_is)
        get_process_context().change_name(my_name_is)

    def __acquire_locked_var(lock_name: Hashable):
        return get_process_context().acquire_semaphore(lock_name)

    def __get_locked_var_value(lock_name: Hashable):
        return get_process_context().get_semaphore(lock_name)

    def __set_locked_var_value(lock_name: Hashable, value):
        get_process_context().set_semaphore(lock_name, value)

    __notify_status("InitCompilation")
    __report_process_name(testcase.testcase_name)
    if not testcase.is_valid:
        return None

    def __dynamic_compilation():
        __notify_status("InitDynCompilation")
        ################################
        # Indirect dynamic compiling parameters
        ################################
        dyn_kernel_name = process_kernel_string("dynamic_op_%s_group%d" % (testcase.op_name, group_id))
        dyn_switch = get_global_storage().dyn_switches
        kernel_meta = get_global_storage().kernel_meta
        device_platform = get_global_storage().device_platform
        device_core_type = get_global_storage().core_type
        os.makedirs(kernel_meta, mode=0o700, exist_ok=True)
        result = DynamicCompilationResult()
        result.kernel_name = dyn_kernel_name
        testcase.dyn_kernel_name = dyn_kernel_name
        ################################
        # Dynamic Shape Compilation
        ################################
        if dyn_switch.enabled:
            if not testcase.dyn_is_valid:
                tiling_result = DynamicOpTilingResult()
                result.tiling_result = tiling_result
                tiling_result.all_set(testcase.dyn_fail_reason)
                result.all_set(testcase.dyn_fail_reason)
                return result
            __notify_status("EstablishInterface")
            interface = OperatorInterface()
            dynamic_characteristic_code = str(testcase.get_compilation_hash())
            if testcase.op_name in interface.special_operator_registry:
                interface.special_operator_registry[testcase.op_name](testcase, "dynamic")
            dynamic_lock_status = __acquire_locked_var(dynamic_characteristic_code)
            if dynamic_lock_status:
                if dyn_switch.realtime:
                    __notify_status("DynCompilation")
                    result.standard_set(*dyn_compile(interface, testcase))
                    # Write extra info into json
                    __notify_status("DynWriteJson")
                    result.write_json(get_global_storage().kernel_meta)
                else:
                    __notify_status("DynManualCompilation")
                    dyn_sch_count = "Unknown"
                    # Find stored compile info
                    compile_info_path = pathlib.Path(pathlib.Path(kernel_meta, "%s.tbetoolkits" % dyn_kernel_name))
                    if compile_info_path.is_file():
                        with open(pathlib.Path(kernel_meta, "%s.tbetoolkits" % dyn_kernel_name),
                                  encoding="UTF-8") as json_file:
                            json_data = json_file.read()
                        json_parsed = json.loads(json_data)
                        compile_info = json_parsed["compile_info"]
                        tiling_op_type = json_parsed["tiling_op_type"]
                        dyn_func_params = json_parsed["dyn_func_params"]
                        kernel_path = pathlib.Path(kernel_meta, "%s" % dyn_kernel_name)
                        # Call ccec
                        dyn_compile_result = tbetoolkits.utilities.cce_manual_compile(str(kernel_path),
                                                                                      device_platform,
                                                                                      device_core_type)
                        dyn_obj_size = "UNKNOWN"
                        dyn_op_pattern = "UNKNOWN"
                        if dyn_compile_result == "SUCC":
                            dyn_obj_size = __get_bin_size(dyn_kernel_name)
                        dyn_compile_time = "-1"
                        result.standard_set(dyn_compile_result,
                                            dyn_compile_time,
                                            compile_info,
                                            tiling_op_type,
                                            dyn_func_params,
                                            dyn_sch_count,
                                            dyn_obj_size,
                                            dyn_op_pattern)
                    else:
                        logging.error("Please run realtime compilation once before using non-realtime mode!")
                        result.all_set("ORIGINAL_INFO_NOT_FOUND")
                logging.debug("Dynamic kernel compilation id %s complete" % dynamic_characteristic_code)
                __set_locked_var_value(dynamic_characteristic_code, result.standard_get())
            else:
                __notify_status("DynWaitCompilation")
                # noinspection PyUnboundLocalVariable
                dyn_lock = __get_locked_var_value(dynamic_characteristic_code)
                while dyn_lock is None:
                    time.sleep(2)
                    dyn_lock = __get_locked_var_value(dynamic_characteristic_code)
                if isinstance(dyn_lock, BaseException):
                    result.all_set(str(dyn_lock.args))
                else:
                    result.standard_set(*dyn_lock)
        else:
            result.all_set("DYN_OFF")
        ################################
        # Dynamic Shape Optiling
        ################################
        __notify_status("OnDynOpTiling")
        tiling_result = DynamicOpTilingResult()
        result.tiling_result = tiling_result
        if result.compile_result != "SUCC":
            tiling_result.all_set(result.compile_result)
        else:
            interface = OperatorInterface()
            tiling_result.standard_set(
                *dyn_do_tiling(interface,
                               testcase,
                               result))
        return result

    def __binary_compilation():
        __notify_status("InitBinCompilation")
        ################################
        # Indirect binary compiling parameters
        ################################
        binary_release_characteristic_code = str(testcase.get_compilation_hash(True))
        bin_kernel_name = "binary_op_%s_%s" % (testcase.op_name, binary_release_characteristic_code.replace("-", "ne"))
        bin_switch = get_global_storage().bin_switches
        kernel_meta = get_global_storage().kernel_meta
        os.makedirs(kernel_meta, mode=0o700, exist_ok=True)
        result = BinaryCompilationResult()
        result.kernel_name = bin_kernel_name
        testcase.bin_kernel_name = bin_kernel_name
        # op_interface read dyn_kernel_name for compilation
        testcase.dyn_kernel_name = bin_kernel_name
        ################################
        # Binary Release Shape Compilation
        ################################
        if bin_switch.enabled:
            if not testcase.dyn_is_valid:
                tiling_result = BinaryOpTilingResult()
                result.tiling_result = tiling_result
                tiling_result.all_set(testcase.dyn_fail_reason)
                result.all_set(testcase.dyn_fail_reason)
                return result
            __notify_status("EstablishInterface")
            interface = OperatorInterface()
            binary_release_lock_status = __acquire_locked_var(binary_release_characteristic_code)
            if binary_release_lock_status:
                if bin_switch.realtime:
                    __notify_status("OnBinCompilation")
                    result.standard_set(*bin_compile(interface, testcase))
                    # Write extra info into json
                    __notify_status("OnBinWriteJson")
                    result.write_json(get_global_storage().kernel_meta)
                else:
                    __notify_status("OnBinManualCompilation")
                    bin_sch_count = "Unknown"
                    # Find stored compile info
                    compile_info_path = pathlib.Path(kernel_meta, "%s.tbetoolkits" % bin_kernel_name)
                    if compile_info_path.is_file():
                        with open(pathlib.Path(kernel_meta, "%s.tbetoolkits" % bin_kernel_name),
                                  encoding="UTF-8") as json_file:
                            json_data = json_file.read()
                        json_parsed = json.loads(json_data)
                        compile_info = json_parsed["compile_info"]
                        tiling_op_type = json_parsed["tiling_op_type"]
                        bin_func_params = json_parsed["bin_func_params"]
                        kernel_path = pathlib.Path(kernel_meta, "%s" % bin_kernel_name)
                        # Call ccec
                        bin_compile_result = \
                            tbetoolkits.utilities.cce_manual_compile(str(kernel_path),
                                                                     get_global_storage().device_platform,
                                                                     get_global_storage().core_type)
                        bin_obj_size = "UNKNOWN"
                        if bin_compile_result == "SUCC":
                            bin_obj_size = __get_bin_size(bin_kernel_name)
                        bin_compile_time = "-1"
                        result.standard_set(bin_compile_result,
                                            bin_compile_time,
                                            compile_info,
                                            tiling_op_type,
                                            bin_func_params,
                                            bin_sch_count,
                                            bin_obj_size)
                    else:
                        logging.error("Please run realtime compilation once before using non-realtime mode!")
                        result.all_set("ORIGINAL_INFO_NOT_FOUND")
                logging.debug("Binary kernel compilation id %s complete" % binary_release_characteristic_code)
                __set_locked_var_value(binary_release_characteristic_code, result.standard_get())
            else:
                __notify_status("OnBinWaitCompilation")
                # noinspection PyUnboundLocalVariable
                bin_lock = __get_locked_var_value(binary_release_characteristic_code)
                while bin_lock is None:
                    time.sleep(2)
                    bin_lock = __get_locked_var_value(binary_release_characteristic_code)
                if isinstance(bin_lock, BaseException):
                    result.all_set(str(bin_lock.args))
                else:
                    result.standard_set(*bin_lock)
        else:
            result.all_set("BIN_OFF")
        ################################
        # Binary Release Shape Optiling
        ################################
        __notify_status("OnBinOpTiling")
        tiling_result = BinaryOpTilingResult()
        result.tiling_result = tiling_result
        if result.compile_result != "SUCC":
            tiling_result.all_set(result.compile_result)
        else:
            interface = OperatorInterface()
            tiling_result.standard_set(
                *dyn_do_tiling(interface,
                               testcase,
                               result))
        return result

    def __static_compilation():
        __notify_status("InitStcCompilation")
        ################################
        # Indirect static compiling parameters
        ################################
        stc_kernel_name = process_kernel_string("static_group%d_%s" % (group_id, testcase.testcase_name))
        stc_switch = get_global_storage().stc_switches
        kernel_meta = get_global_storage().kernel_meta
        device_platform = get_global_storage().device_platform
        device_core_type = get_global_storage().core_type
        os.makedirs(kernel_meta, mode=0o700, exist_ok=True)
        result = StaticCompilationResult()
        result.stc_kernel_name = stc_kernel_name
        testcase.stc_kernel_name = stc_kernel_name
        __notify_status("EstablishInterface")
        interface = OperatorInterface()
        ################################
        # Static Shape Compilation
        ################################
        if stc_switch.enabled:
            if stc_switch.realtime:
                __notify_status("OnStcCompilation")
                result.standard_set(*stc_compile(interface, testcase))
            else:
                __notify_status("OnStcManualCompilation")
                # Call ccec
                kernel_path = pathlib.Path(kernel_meta, "%s" % stc_kernel_name)
                stc_compile_result = tbetoolkits.utilities.cce_manual_compile(str(kernel_path),
                                                                              device_platform,
                                                                              device_core_type)
                stc_op_pattern = "Unknown"
                if stc_compile_result == "SUCC":
                    # Get Info
                    stc_block_dim, stc_workspaces = tbetoolkits.utilities.get_stc_json_op_data(kernel_path)
                    stc_compile_time = "UNKNOWN"
                    rl_query_result = "Manual"
                else:
                    stc_block_dim, stc_workspaces = 0, ()
                    stc_compile_time = stc_compile_result
                    rl_query_result = "Manual"
                result.standard_set(stc_compile_result, stc_compile_time, stc_block_dim, stc_workspaces,
                                    rl_query_result, stc_op_pattern)
        else:
            result.all_set("STC_OFF")
        return result

    def __const_compilation():
        __notify_status("InitCstCompilation")
        ################################
        # Indirect static compiling parameters
        ################################
        cst_kernel_name = process_kernel_string("const_group%d_%s" % (group_id, testcase.testcase_name))
        cst_switch = get_global_storage().cst_switches
        kernel_meta = get_global_storage().kernel_meta
        device_core_type = get_global_storage().core_type
        device_platform = get_global_storage().device_platform
        os.makedirs(kernel_meta, mode=0o700, exist_ok=True)
        result = ConstCompilationResult()
        result.cst_kernel_name = cst_kernel_name
        testcase.cst_kernel_name = cst_kernel_name
        testcase.dyn_kernel_name = cst_kernel_name
        ################################
        # Const Shape Compilation
        ################################
        if cst_switch.enabled:
            if not testcase.dyn_is_valid:
                result.all_set(testcase.dyn_fail_reason)
                return result
            __notify_status("EstablishInterface")
            interface = OperatorInterface()
            if cst_switch.realtime:
                __notify_status("OnCstCompilation")
                result.standard_set(*cst_compile(interface, testcase))
                result.write_json(get_global_storage().kernel_meta)
            else:
                __notify_status("OnCstManualCompilation")
                # Find stored compile info
                compile_info_path = pathlib.Path(kernel_meta, "%s.tbetoolkits" % cst_kernel_name)
                cst_op_pattern = "UNKNOWN"
                if compile_info_path.is_file():
                    with open(pathlib.Path(kernel_meta, "%s.tbetoolkits" % cst_kernel_name),
                              encoding="UTF-8") as json_file:
                        json_data = json_file.read()
                    json_parsed = json.loads(json_data)
                    cst_func_params = json_parsed["cst_func_params"]
                    cst_rl_status = json_parsed["cst_rl_status"]
                    # Call ccec
                    kernel_path = pathlib.Path(kernel_meta, "%s" % cst_kernel_name)
                    cst_compile_result = tbetoolkits.utilities.cce_manual_compile(str(kernel_path),
                                                                                  device_platform,
                                                                                  device_core_type)
                    if cst_compile_result == "SUCC":
                        # Get Info
                        cst_block_dim, cst_workspaces = tbetoolkits.utilities.get_stc_json_op_data(kernel_path)
                        stc_compile_time = "UNKNOWN"
                    else:
                        cst_block_dim, cst_workspaces = 0, ()
                        stc_compile_time = cst_compile_result
                        cst_func_params = (cst_compile_result,)
                else:
                    logging.error("Please run realtime compilation once before using non-realtime mode!")
                    cst_compile_result = "ORIGINAL_INFO_NOT_FOUND"
                    cst_block_dim, cst_workspaces = 0, ()
                    stc_compile_time = cst_compile_result
                    cst_func_params = (cst_compile_result,)
                    cst_rl_status = cst_compile_result
                result.standard_set(cst_compile_result, stc_compile_time, cst_block_dim, cst_workspaces,
                                    cst_func_params, cst_op_pattern, cst_rl_status)
        else:
            result.all_set("CST_OFF")
        return result

    if mode == "dynamic":
        return __dynamic_compilation()
    if mode == "static":
        return __static_compilation()
    if mode == "const":
        return __const_compilation()
    if mode == "binary":
        return __binary_compilation()
    raise RuntimeError("Unknown mode %s" % mode)


def __get_bin_size(kernel_name: str) -> str:
    kernel_meta = get_global_storage().kernel_meta
    return str(pathlib.Path(kernel_meta, "%s.o" % kernel_name).stat().st_size)


def __get_kernel_size(kernel_name: str, tiling_key_index: Union[str, int]) -> str:
    return "UNKNOWN"


def dyn_compile(interface: OperatorInterface,
                testcase: UniversalTestcaseStructure) -> Tuple[str, str, dict, str, Tuple[str], str, str, str]:
    """
    Wrapper function for op_interface dynamic operator compilation sequence
    """
    # noinspection PyBroadException
    try:
        dyn_operator_params = (eliminate_scalar_shapes(testcase.dyn_inputs),
                               eliminate_scalar_shapes(testcase.dyn_ori_inputs),
                               testcase.dyn_input_dtypes,
                               testcase.dyn_input_formats,
                               testcase.dyn_input_ori_formats,
                               testcase.dyn_input_ranges,
                               testcase.dyn_outputs,
                               testcase.dyn_ori_outputs,
                               testcase.output_dtypes,
                               testcase.output_formats,
                               testcase.output_ori_formats,
                               testcase.dyn_output_ranges)
        dyn_operator_params = interface.prepare_operator_parameters(*dyn_operator_params)
        compile_result = \
            interface.compile_dynamic_shape(dyn_operator_params, testcase)
        if compile_result:
            tiling_op_type, compile_info, compile_time, dyn_func_params, sch_count, op_pattern, _ = compile_result
            dyn_compile_result = "SUCC"
            dyn_compile_time = compile_time
            try:
                dyn_obj_size = __get_bin_size(testcase.dyn_kernel_name)
            except FileNotFoundError:
                dyn_compile_result = "DYN_OPERATOR_BUILD_LOST"
                dyn_obj_size = dyn_compile_result
            else:
                logging.debug("Compilation of dynamic kernel %s success, received compile info:\n%s"
                              % (testcase.dyn_kernel_name,
                                 json.dumps(compile_info)))
        else:
            fail_reason = "DYN_OPERATOR_NOT_FOUND"
            dyn_compile_result = fail_reason
            dyn_compile_time = fail_reason
            compile_info = {}
            tiling_op_type = fail_reason
            dyn_func_params = None
            sch_count = fail_reason
            dyn_obj_size = fail_reason
            op_pattern = fail_reason
            logging.error(f"Dynamic operator {testcase.dyn_kernel_name} not found")
    except:
        fail_reason = "DYN_COMPILE_FAILURE"
        dyn_compile_result = fail_reason
        dyn_compile_time = fail_reason
        compile_info = {}
        tiling_op_type = fail_reason
        dyn_func_params = None
        sch_count = fail_reason
        dyn_obj_size = fail_reason
        op_pattern = fail_reason
        logging.exception("Compilation of dynamic operator: %s failed" % testcase.dyn_kernel_name)
    return (dyn_compile_result, dyn_compile_time, compile_info, tiling_op_type,
            dyn_func_params, sch_count, dyn_obj_size, op_pattern)


def bin_compile(interface: OperatorInterface,
                testcase: UniversalTestcaseStructure) -> Tuple[str, str, dict, str, Tuple[str], str, str]:
    """
    Wrapper function for op_interface binary operator compilation sequence
    """
    # noinspection PyBroadException
    try:
        bin_inputs = []
        bin_ranges = []
        for idx, dyn_input in enumerate(testcase.dyn_inputs):
            if idx in testcase.const_input_indexes or dyn_input is None:
                bin_inputs.append(dyn_input)
                bin_ranges.append(get(testcase.dyn_input_ranges, idx, (None, None)))
            else:
                bin_inputs.append((-2,))
                bin_ranges.append(((1, None),))
        bin_inputs = tuple(bin_inputs)
        bin_op_params = (eliminate_scalar_shapes(bin_inputs),
                         eliminate_scalar_shapes(bin_inputs),
                         testcase.dyn_input_dtypes,
                         testcase.dyn_input_formats,
                         testcase.dyn_input_ori_formats,
                         bin_ranges,
                         len(testcase.dyn_outputs) * ((-2,),),
                         len(testcase.dyn_outputs) * ((-2,),),
                         testcase.output_dtypes,
                         testcase.output_formats,
                         testcase.output_ori_formats,
                         len(testcase.dyn_outputs) * (((1, None),),))
        bin_operator_params = interface.prepare_operator_parameters(*bin_op_params)
        compile_result = \
            interface.compile_dynamic_shape(bin_operator_params, testcase, mode="Binary")
        if compile_result:
            tiling_op_type, compile_info, compile_time, bin_func_params, sch_count, bin_op_pattern, _ = compile_result
            bin_compile_result = "SUCC"
            bin_compile_time = compile_time
            bin_obj_size = __get_bin_size(testcase.bin_kernel_name)
            logging.debug("Compilation of binary kernel %s success, received compile info:\n%s"
                          % (testcase.bin_kernel_name,
                             json.dumps(compile_info)))
        else:
            fail_reason = "BIN_OPERATOR_NOT_FOUND"
            bin_compile_result = fail_reason
            bin_compile_time = fail_reason
            compile_info = {}
            tiling_op_type = fail_reason
            bin_func_params = None
            sch_count = fail_reason
            bin_obj_size = fail_reason
            logging.error("Binary operator %s not found" % testcase.bin_kernel_name)
    except:
        fail_reason = "BIN_COMPILE_FAILURE"
        bin_compile_result = fail_reason
        bin_compile_time = fail_reason
        compile_info = {}
        tiling_op_type = fail_reason
        bin_func_params = None
        sch_count = fail_reason
        bin_obj_size = fail_reason
        logging.exception("Compilation of binary operator: %s failed" % testcase.bin_kernel_name)
    return bin_compile_result, bin_compile_time, compile_info, tiling_op_type, bin_func_params, sch_count, bin_obj_size


def cst_compile(interface: OperatorInterface,
                testcase: UniversalTestcaseStructure) -> Tuple[str, str, int, tuple, tuple, str, str]:
    """
    Wrapper function for op_interface const operator compilation sequence
    """
    # noinspection PyBroadException
    try:
        kernel_meta = get_global_storage().kernel_meta
        cst_operator_params = interface.prepare_operator_parameters_const(testcase)
        compile_result = \
            interface.compile_dynamic_shape(cst_operator_params, testcase, True, "Const")
        if compile_result:
            tiling_op_type, compile_info, compile_time, dyn_func_params, _, dyn_op_pattern, cst_rl_status = \
                compile_result
            cst_compile_result = "SUCC"
            cst_compile_time = str(compile_time)
            try:
                cst_block_dim, cst_workspaces = get_stc_json_op_data(pathlib.Path(kernel_meta,
                                                                                  testcase.cst_kernel_name))
            except FileNotFoundError:
                fail_reason = "CST_OPERATOR_BUILD_LOST"
                cst_rl_status = fail_reason
                cst_compile_result = fail_reason
                cst_compile_time = fail_reason
                cst_block_dim = 0
                dyn_func_params = None
                cst_workspaces = ()
                dyn_op_pattern = fail_reason
                logging.error(f"Const operator {testcase.cst_kernel_name} build artifacts not found")
        else:
            fail_reason = "CST_OPERATOR_NOT_FOUND"
            cst_rl_status = fail_reason
            cst_compile_result = fail_reason
            cst_compile_time = fail_reason
            cst_block_dim = 0
            dyn_func_params = None
            cst_workspaces = ()
            dyn_op_pattern = fail_reason
            logging.error("Const operator %s not found" % testcase.cst_kernel_name)
    except:
        fail_reason = "CST_COMPILE_FAILURE"
        cst_rl_status = fail_reason
        cst_compile_result = fail_reason
        cst_compile_time = fail_reason
        cst_block_dim = 0
        dyn_func_params = None
        cst_workspaces = ()
        dyn_op_pattern = fail_reason
        logging.exception("Compilation of const operator: %s failed"
                          % testcase.cst_kernel_name)

    return (cst_compile_result, cst_compile_time, cst_block_dim, cst_workspaces,
            dyn_func_params, dyn_op_pattern, cst_rl_status)


def stc_compile(interface: OperatorInterface,
                testcase: UniversalTestcaseStructure) -> Tuple[str, str, int, tuple, str, str]:
    """
    Wrapper function for op_interface dynamic operator compilation sequence
    """
    # noinspection PyBroadException
    try:
        stc_op_params = (eliminate_scalar_shapes(testcase.stc_inputs),
                         eliminate_scalar_shapes(testcase.stc_ori_inputs),
                         testcase.stc_input_dtypes,
                         testcase.stc_input_formats,
                         testcase.stc_input_ori_formats,
                         (None,),
                         testcase.stc_outputs,
                         testcase.stc_ori_outputs,
                         testcase.output_dtypes,
                         testcase.output_formats,
                         testcase.output_ori_formats,
                         (None,))
        kernel_meta = get_global_storage().kernel_meta
        stc_operator_params = interface.prepare_operator_parameters(*stc_op_params)
        compile_result = \
            interface.compile_static_shape(stc_operator_params,
                                           testcase)
        if compile_result:
            compile_time, rl_query_result, stc_op_pattern = compile_result
            stc_compile_result = "SUCC"
            stc_compile_time = str(compile_time)
            try:
                stc_block_dim, stc_workspaces = get_stc_json_op_data(pathlib.Path(kernel_meta,
                                                                                  testcase.stc_kernel_name))
            except FileNotFoundError:
                fail_reason = "STC_OPERATOR_BUILD_LOST"
                stc_compile_result = fail_reason
                stc_compile_time = fail_reason
                rl_query_result = fail_reason
                stc_block_dim = fail_reason
                stc_op_pattern = fail_reason
                stc_workspaces = ()
                logging.error(f"Static operator {testcase.stc_kernel_name} build artifacts not found")
        else:
            fail_reason = "STC_OPERATOR_NOT_FOUND"
            stc_compile_result = fail_reason
            stc_compile_time = fail_reason
            rl_query_result = fail_reason
            stc_block_dim = fail_reason
            stc_op_pattern = fail_reason
            stc_workspaces = ()
            logging.error(f"Static operator {testcase.stc_kernel_name} not found")
    except:
        fail_reason = "STC_COMPILE_FAILURE"
        stc_compile_result = fail_reason
        stc_compile_time = fail_reason
        rl_query_result = fail_reason
        stc_block_dim = fail_reason
        stc_op_pattern = fail_reason
        stc_workspaces = ()
        logging.exception(f"Compilation of static operator: {testcase.stc_kernel_name} failed")
    return stc_compile_result, stc_compile_time, stc_block_dim, stc_workspaces, rl_query_result, stc_op_pattern


def dyn_do_tiling(interface: OperatorInterface,
                  testcase: UniversalTestcaseStructure,
                  result: Union[DynamicCompilationResult,
                                BinaryCompilationResult]) -> Tuple[int, int, bytes, tuple, str, str]:
    """
    Wrapper function for op_interface dynamic operator op_tiling sequence
    """
    # noinspection PyBroadException
    try:
        tiling_result = interface.call_const_op_tiling(result,
                                                       testcase)
    except BaseException as e:
        logging.exception("OPTILING_FAILURE")
        dyn_block_dim = 0
        tiling_data = str(e.args).encode("UTF-8")
        workspaces = ()
        tiling_time = "UNKNOWN"
        dyn_kernel_size = "UNKNOWN"
        tiling_key = -404
    else:
        dyn_block_dim = int(tiling_result["block_dim"]) if tiling_result["block_dim"] > 0 else 0
        tiling_data = tiling_result["tiling_data"]
        tiling_key = int(numpy.uint32(tiling_result["tiling_key"]))
        workspaces = tuple(tiling_result["workspaces"])
        tiling_time = tiling_result["tiling_time"]
        tiling_key_index = "UNKNOWN"
        if "_vars" in result.compile_info:
            if tiling_key in result.compile_info["_vars"] or str(tiling_key) in result.compile_info["_vars"]:
                if str(tiling_key) in result.compile_info["_vars"]:
                    tiling_key_index = tuple(result.compile_info["_vars"].keys()).index(str(tiling_key))
                elif tiling_key in result.compile_info["_vars"]:
                    tiling_key_index = tuple(result.compile_info["_vars"].keys()).index(tiling_key)
                else:
                    tiling_key_index = "UNKNOWN"
        dyn_kernel_size = __get_kernel_size(result.kernel_name, tiling_key_index)

    return dyn_block_dim, tiling_key, tiling_data, workspaces, tiling_time, dyn_kernel_size
