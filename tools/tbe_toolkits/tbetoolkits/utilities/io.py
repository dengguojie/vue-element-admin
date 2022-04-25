#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Precious Utility Functions
"""
# Standard Packages
import os
import sys
import urllib.request
import json
import ctypes
import pathlib
import logging
import subprocess
from functools import wraps
from typing import Tuple
from typing import Sequence

# Third-Party Packages
from .mappings import get_arch_for_ccec
from .classes import PMU_MODE
from .classes import MODE
from .classes import SWITCHES

param_map = {}
param_help = {}


def read_file(file_path: str, size_limit: int = 1024 * 1024 * 1024) -> bytes:
    """
    :param file_path: Path to the file
    :param size_limit: Raise an Exception if the file is too large
    :return: binary object
    """
    file_size = os.stat(file_path).st_size
    if file_size > size_limit:
        raise IOError("File is too large! Size of %s exceeds the limit: %s"
                      % (file_path, size_limit))
    with open(file_path, "rb") as file:
        file_content = file.read()
    return file_content


def cce_manual_compile(dyn_kernel_name: str, platform: str, core_type: str) -> str:
    """
    CCEC Manual Compilation
    :param dyn_kernel_name:
    :param platform:
    :param core_type:
    :return:
    """
    if not pathlib.Path("%s.cce" % dyn_kernel_name).is_file():
        if pathlib.Path("%s.o" % dyn_kernel_name).is_file():
            return "SUCC"
        else:
            return "FOUND_NOTHING"
    ccec = subprocess.Popen(["ccec", "-O2", "%s.cce" % dyn_kernel_name,
                             "--cce-aicore-arch=%s" % get_arch_for_ccec(platform, core_type),
                             "--cce-aicore-only", "-o", "%s.i" % dyn_kernel_name,
                             "-mllvm", "-cce-aicore-function-stack-size=16000",
                             "-mllvm", "-cce-aicore-record-overflow=true",
                             "-mllvm", "--cce-aicore-jump-expand=true",
                             "-mllvm", "-cce-aicore-addr-transform"],
                            bufsize=0,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    ccec_return = ccec.communicate()[1].decode("UTF-8")
    if not ccec.returncode == 0:
        logging.error("CCEC Compilation Failure %d: %s" % (ccec.returncode, ccec_return))
        return ccec_return
    ld = subprocess.Popen(["ld.lld", "-m", "aicorelinux",
                           "-Ttext=0", "%s.i" % dyn_kernel_name,
                           "-static", "-o",
                           "%s.o" % dyn_kernel_name],
                          bufsize=0,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    ld_return = ld.communicate()[1].decode("UTF-8")
    if not ld.returncode == 0:
        logging.error("CCEC-LD-LLD Compilation Failure %d: %s" % (ld.returncode, ld_return))
        return ld_return
    return "SUCC"


def get_stc_json_op_data(stc_kernel_name) -> Tuple[int, tuple]:
    """
    Get static operator compile info from json
    :param stc_kernel_name:
    :return:
    """
    with open("%s.json" % stc_kernel_name, encoding="UTF-8") as f:
        raw_json_data = f.read()
        # noinspection PyBroadException
        try:
            json_data = json.loads(raw_json_data)
        except:
            logging.exception("Json read failure, received json:\n%s" % raw_json_data)
            raise
    stc_block_dim = json_data["blockDim"]
    stc_workspaces = tuple(json_data["workspace"]["size"]) if "workspace" in json_data else ()
    return int(stc_block_dim), stc_workspaces


def get_loaded_so_path(loaded_cdll: ctypes.CDLL) -> str:
    """
    :param loaded_cdll:
    :return:
    """
    # noinspection PyProtectedMember
    results = [o.split(" ")[-1] for o in
               subprocess.check_output(["lsof", "-p", str(os.getpid())], encoding="UTF-8").split("\n")
               if loaded_cdll._name in o]
    return '|'.join(results) if len(results) > 0 else "UNKNOWN"


def download_file(url: str, save_path: str):
    """Download file from web"""
    urllib.request.urlretrieve(url, save_path)


def register_param(param_names: Sequence[str], param_help_message: str = None):
    """Register postprocessing function"""

    def __inner_param_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for param_name in param_names:
            if param_name in param_map:
                logging.warning("param function of %s has already been registered!" % param_name)
            param_map[param_name] = __wrapper
            param_help[param_name] = param_help_message
        return __wrapper

    return __inner_param_registry


def parse_params(switches: SWITCHES, params: Sequence):
    """
    :param switches:
    :param params:
    :return:
    """
    param_invalid = False
    for param in params:
        param = param.strip()
        main_param: str = param.split("=")[0]
        main_param: str = main_param.lower()
        if main_param in param_map:
            secondary_param = None
            if "=" in param:
                secondary_param = "=".join(param.split("=")[1:])
            param_map[main_param](switches, secondary_param)
        elif not param.startswith("-"):
            if switches.input_file_name is None:
                switches.input_file_name = param
            elif switches.output_file_name is None:
                switches.output_file_name = param
            else:
                logging.error(f"Invalid param: {param}")
                param_invalid = True
        else:
            logging.error(f"Invalid param: {param}")
            param_invalid = True
    if param_invalid:
        sys.exit(-1)


@register_param(["--dynamic", "-d"], "此选项控制动态Shape测试流程的开关，默认值为true 用法：\n"
                                     "dynamic=false 或 -d=false 即可关闭动态Shape测试流程")
def __set_dynamic(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.dyn_switches.enabled = True
    elif secondary_param.lower() == "false":
        switches.dyn_switches.enabled = False
    else:
        raise RuntimeError("Invalid dynamic shape mode: %s" % secondary_param)


@register_param(["--dynamic-run", "--dr"])
def __set_dynamic(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.dyn_switches.prof = True
    elif secondary_param.lower() == "false":
        switches.dyn_switches.prof = False
    else:
        raise RuntimeError("Invalid dynamic shape profiling mode: %s" % secondary_param)


@register_param(["--static", "-s"])
def __set_static(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.stc_switches.enabled = True
    elif secondary_param.lower() == "false":
        switches.stc_switches.enabled = False
    else:
        raise RuntimeError("Invalid static shape mode: %s" % secondary_param)


@register_param(["--static-profiling", "--sp"])
def __set_static_prof(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.stc_switches.prof = True
    elif secondary_param.lower() == "false":
        switches.stc_switches.prof = False
    else:
        raise RuntimeError("Invalid static shape profiling mode: %s" % secondary_param)


@register_param(["--const", "-c"])
def __set_const(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.cst_switches.enabled = True
    elif secondary_param.lower() == "false":
        switches.cst_switches.enabled = False
    else:
        raise RuntimeError("Invalid const shape mode: %s" % secondary_param)


@register_param(["--const-profiling", "--cp"])
def __set_const(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.cst_switches.prof = True
    elif secondary_param.lower() == "false":
        switches.cst_switches.prof = False
    else:
        raise RuntimeError("Invalid const shape profiling mode: %s" % secondary_param)


@register_param(["--binary", "-b"])
def __set_binary(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.bin_switches.enabled = True
    elif secondary_param.lower() == "false":
        switches.bin_switches.enabled = False
    else:
        raise RuntimeError("Invalid binary release shape mode: %s" % secondary_param)


@register_param(["--binary-profiling", "--bp"])
def __set_binary(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.bin_switches.prof = True
    elif secondary_param.lower() == "false":
        switches.bin_switches.prof = False
    else:
        raise RuntimeError("Invalid binary release shape profiling mode: %s" % secondary_param)


@register_param(["--compile-only", "--compile", "--co"])
def __set_compile_only(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.dyn_switches.prof = False
        switches.stc_switches.prof = False
        switches.cst_switches.prof = False
        switches.bin_switches.prof = False
    elif secondary_param.lower() == "false":
        switches.dyn_switches.prof = True
        switches.stc_switches.prof = True
        switches.cst_switches.prof = True
        switches.bin_switches.prof = True
    else:
        raise RuntimeError("Invalid compile only mode: %s" % secondary_param)


@register_param(["--egg"])
def __easter_egg(*_, **__):
    print("The quick brown fox jumps over the lazy dog")
    sys.exit(0)


@register_param(["--tiling-run", "--tr"])
def __set_hbm_size_limit(switches: SWITCHES, secondary_param: str):
    switches.tiling_run_time = int(secondary_param)


@register_param(["--gpu", "-g"])
def __set_gpu_mode(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "tensorflow":
        switches.mode = MODE.GPU_TENSORFLOW
    elif secondary_param.lower() == "torch":
        switches.mode = MODE.GPU_PYTORCH
    else:
        raise RuntimeError("Invalid gpu mode: %s" % secondary_param)


@register_param(["--limit", "-l"])
def __set_hbm_size_limit(switches: SWITCHES, secondary_param: str):
    switches.DAVINCI_HBM_SIZE_LIMIT = int(secondary_param)


@register_param(["--testcase", "-t"])
def __set_testcase(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify name for testcase")
    else:
        switches.selected_testcases = secondary_param.split(",")


@register_param(["--device"])
def __set_device_num(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        switches.device_count = -1
    else:
        switches.device_count = int(secondary_param)


@register_param(["--device-blacklist"])
def __set_device_blacklist(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Device blacklist should be a list of device id, such as: 1,2,3,4,5,6,7")
    else:
        switches.device_blacklist = tuple(int(dev_id) for dev_id in secondary_param.split(','))


@register_param(["--fatbin-parallel", "--fp"])
def __set_parallel_fatbin(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.parallel_fatbin = True
    elif secondary_param.lower() == "false":
        switches.parallel_fatbin = False
    else:
        raise RuntimeError("Invalid parallel fatbin mode: %s" % secondary_param)


@register_param(["--pmu"])
def __enable_pmu(switches: SWITCHES, secondary_param: str):
    switches.PMU = True
    if secondary_param == "0":
        switches.PMU_MODE = PMU_MODE.DEFAULT
    if secondary_param == "1":
        switches.PMU_MODE = PMU_MODE.ADVANCED


@register_param(["--csv-preserve"])
def __enable_shape_int64(switches: SWITCHES, _: str):
    switches.preserve_original_csv = True


@register_param(["--shape64"])
def __enable_shape_int64(switches: SWITCHES, _: str):
    switches.int64_shape_mode = True


@register_param(["--ascend910a_model"])
def __enable_ascend910a_model(switches: SWITCHES, secondary_param: str):
    switches.device_platform = "Ascend910A"
    switches.core_type = "AiCore"
    if secondary_param is None:
        switches.mode = MODE.ASCEND_PEMMODEL
    elif secondary_param.lower() == "pem":
        switches.mode = MODE.ASCEND_PEMMODEL
    elif secondary_param.lower() == "ca":
        switches.mode = MODE.ASCEND_CAMODEL
    else:
        raise ValueError("Unknown model mode: %s" % secondary_param)


@register_param(["--single-log"])
def __single_log_mode(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() in ("true", "1"):
        switches.single_testcase_log_mode = True
    elif secondary_param.lower() == ("false", "0"):
        switches.single_testcase_log_mode = False
    else:
        raise RuntimeError("Invalid single testcase log mode switch: %s" % secondary_param)


@register_param(["--platform", "--plat"])
def __set_device_platform(switches: SWITCHES, secondary_param: str):
    switches.device_platform = secondary_param


@register_param(["--core-type", "--ct"])
def __set_core_type(switches: SWITCHES, secondary_param: str):
    switches.core_type = secondary_param


@register_param(["--run"])
def __set_run_time(switches: SWITCHES, secondary_param: str):
    try:
        run_time: int = int(secondary_param)
    except:
        raise RuntimeError("Specified run time is not valid: %s" % secondary_param)
    else:
        switches.run_time = run_time


@register_param(["--perf-relative-tolerance", "--prt"])
def __set_perf_relative_tolerance(switches: SWITCHES, secondary_param: str):
    switches.perf_threshold = (eval(secondary_param), switches.perf_threshold[1])


@register_param(["--perf-absolute-tolerance", "--pat"])
def __set_perf_absolute_tolerance(switches: SWITCHES, secondary_param: str):
    switches.perf_threshold = (switches.perf_threshold[0], eval(secondary_param))


@register_param(["--debug"])
def __set_debug(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.single_case_debugging = True
    elif secondary_param.lower() == "false":
        switches.single_case_debugging = False
    else:
        raise RuntimeError("Invalid debugging mode: %s" % secondary_param)


@register_param(["--model-target", "--mt"])
def __set_model_target_block(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify indexes for model target block")
    else:
        switches.model_target_block_dim = tuple(map(int, secondary_param.split(",")))


@register_param(["--ti", "--testcase-index"])
def __set_testcase_indexes(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify indexes for testcases")
    else:
        selected_indexes = []
        indexes = secondary_param.split(",")
        for _i in indexes:
            if '-' in _i:
                lower = int(_i.split('-')[0])
                higher = int(_i.split('-')[1])
                selected_indexes += list(range(lower, higher))
            else:
                selected_indexes.append(int(_i))
        switches.selected_testcase_indexes = tuple(selected_indexes)


@register_param(["--op", "--operator"])
def __set_testcase_op_name(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify op_name for testcases")
    else:
        switches.selected_operators = tuple(map(str, secondary_param.split(",")))


@register_param(["--pc", "--process-count"])
def __set_process_count(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify process count for each device")
    else:
        switches.process_per_device = int(secondary_param)


@register_param(["--tc", "--testcase-count"])
def __set_testcase_count(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify testcase count")
    else:
        switches.selected_testcase_count = int(secondary_param)


@register_param(["--server"])
def __testcase_result_server(switches: SWITCHES, secondary_param: str):
    switches.testcase_server = secondary_param


@register_param(["--pm", "--precision-mode"])
def __precision_mode(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.do_precision_test = True
    elif secondary_param.lower() == "false":
        switches.do_precision_test = False
    else:
        raise RuntimeError("Invalid Precision Test Mode: %s" % secondary_param)


@register_param(["--print"])
def __precision_mode(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.summary_print = True
    elif secondary_param.lower() == "false":
        switches.summary_print = False
    else:
        raise RuntimeError("Invalid Summary Print Mode: %s" % secondary_param)


@register_param(["--ub", "--override-ub-size"])
def __soc_ub_size_override(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Could not override UB_SIZE to None")
    else:
        switches.soc_spec_override["UB_SIZE"] = int(secondary_param)


@register_param(["--aicore", "--override-core-num"])
def __soc_core_num_override(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Could not override CORE_NUM to None")
    else:
        switches.soc_spec_override["CORE_NUM"] = int(secondary_param)
