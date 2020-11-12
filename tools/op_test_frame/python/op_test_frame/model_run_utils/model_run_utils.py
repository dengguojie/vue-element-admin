# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
model run utils
"""
import os
import re
import stat
import ctypes
from ctypes import Structure
from ctypes import c_int
from ctypes import c_char_p
from functools import reduce as func_reduce
from multiprocessing import Pool
from op_test_frame.utils import file_util

SO_NAME = "libmodel_run_tool.so"

CAMODEL_LOG_PATH_ENV = "CAMODEL_LOG_PATH"
LD_LIBRARY_PATH_ENV = "LD_LIBRARY_PATH"
DATA_DIR_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP

RUN_MODE = None
MODEL_SOC_VERSION = None
SIMULATOR_PATH = None
OP_RUN_UTILS = None
SIMULATOR_SO_HANDLERS = []
MODE_SETTED = False

SUPPORT_MODEL_LIST = ["pv", "ca", "tm"]
MODEL_SO_LIST_MAP = {
    "pv": ("lib_pvmodel.so", "libtsch.so", "libnpu_drv_pvmodel.so", "libruntime_cmodel.so"),
    "ca": (
        "libcamodel.so", "libtsch_camodel.so", "libnpu_drv_camodel.so", "libruntime_camodel.so"),
    "tm": (
        "libpem_davinci.so", "libcamodel.so", "libtsch_camodel.so", "libnpu_drv_camodel.so", "libruntime_camodel.so"),
}


def set_run_mode(mode, soc_version, simulator_dir):
    """
    set run mode
    :param mode: can be "pv", "tm"
    :param soc_version: soc verison
    :param simulator_dir: simulator dir
    :return: None
    """
    global RUN_MODE  # pylint: disable=global-statement
    global SIMULATOR_PATH  # pylint: disable=global-statement
    global MODEL_SOC_VERSION  # pylint: disable=global-statement
    global MODE_SETTED  # pylint: disable=global-statement
    if MODE_SETTED:
        if mode not in MODEL_SO_LIST_MAP.keys():
            raise RuntimeError(
                "Not support mode: %s, supported mode is [%s]." % (mode, ",".join(MODEL_SO_LIST_MAP.keys())))

        if RUN_MODE != mode:
            raise RuntimeError(
                "Only support run one mode in process, current run mode is %s, try to set mode %s failed." % (
                    RUN_MODE, mode))

        if MODEL_SOC_VERSION != soc_version:
            raise RuntimeError("Only support run one soc version, current soc version: %s, try run soc version: %s" % (
                MODEL_SOC_VERSION, soc_version))

    else:
        simulator_dir = os.path.realpath(simulator_dir)
        simulator_dir_soc_path = os.path.join(simulator_dir, soc_version, "lib")
        load_simulator_so(mode, simulator_dir_soc_path)
        os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":" + simulator_dir_soc_path
        common_data_path = os.path.join(simulator_dir, "common", "data")
        os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":" + common_data_path
        MODE_SETTED = True
        MODEL_SOC_VERSION = soc_version
        SIMULATOR_PATH = simulator_dir
        RUN_MODE = mode


class OpPrams(Structure):  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """
    op params
    """
    _fields_ = [("inputCnt", c_int),
                ("outputCnt", c_int),
                ("inputSizes", c_char_p),
                ("outputSizes", c_char_p),
                ("inputDataPaths", c_char_p),
                ("outputDataPaths", c_char_p),
                ("binFilePath", c_char_p),
                ("kernelFuncName", c_char_p),
                ("jsonFilePath", c_char_p), ]


def load_simulator_so(run_mode_tmp, simulator_dir):
    """
    load simulator so
    :param run_mode_tmp: run_mode
    :param simulator_dir: simulator directory
    :return: None
    """
    so_list = MODEL_SO_LIST_MAP[run_mode_tmp]
    for so_name in so_list:
        so_path = os.path.join(simulator_dir, so_name)
        so_handler = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        SIMULATOR_SO_HANDLERS.append(so_handler)


def load_model_run_tool_so():
    """
    load model run tool so
    :return: None
    """
    global OP_RUN_UTILS  # pylint: disable=global-statement
    if not MODE_SETTED:
        raise RuntimeError(
            "Not set run model, please call set_run_mode(mode, soc_version, simulator_dir), "
            "you may not set simulator mode or simulator lib path")
    if OP_RUN_UTILS:
        # already loaded
        return
    lib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "libs")
    so_path = os.path.join(lib_path, SO_NAME)
    if os.path.isfile(so_path):
        OP_RUN_UTILS = ctypes.CDLL(so_path, mode=ctypes.RTLD_LOCAL)
        return
    if os.environ.get(LD_LIBRARY_PATH_ENV):
        all_ld_paths = os.environ[LD_LIBRARY_PATH_ENV].split(":")
        for one_path in all_ld_paths:
            so_path = os.path.join(one_path, SO_NAME)
            if os.path.isfile(so_path):
                OP_RUN_UTILS = ctypes.cdll.LoadLibrary(so_path)
                break

    if not OP_RUN_UTILS:
        raise RuntimeError("Not found %s in LD_LIBRARY_PATH, please check your configuration." % SO_NAME)
    if OP_RUN_UTILS:
        OP_RUN_UTILS.RunOp.argtypes = [OpPrams]


def calc_input_size(shape, dtype):
    """
    calc input size
    :param shape: shape
    :param dtype: dtype
    :return: input size
    """
    bit_cnt = int(re.findall(r"\d+", dtype)[0])
    byte_cnt = bit_cnt / 8
    shape_size = func_reduce(lambda x, y: x * y, shape)
    return int(shape_size * byte_cnt)


def check_op_params(op_param):
    """
    check op param
    :param op_param: op param
    :return: True or False
    """
    op_param_info_keys = op_param.keys()

    if "binFilePath" not in op_param_info_keys:
        print("[ERROR] this op param has no bin file path, can't run this op")
        return False
    if "kernelFuncName" not in op_param_info_keys:
        print("[ERROR] this op param has no kernel func name, can't run this op")
        return False
    if "jsonFilePath" not in op_param_info_keys:
        print("[ERROR] this op param has no json file path, can't run this op")
        return False
    return True


def build_model_run_params(op_param):
    """
    build model run params
    :param op_param: op param
    :return: run param
    """
    # check input count match input shape and data path info
    op_param_info_keys = op_param.keys()
    input_infos = []
    output_infos = []
    if "inputInfos" in op_param_info_keys:
        input_infos = op_param["inputInfos"]
    if "outputInfos" in op_param_info_keys:
        output_infos = op_param["outputInfos"]
    if not input_infos:
        print("[Warning] this op param has no input info")
    if not output_infos:
        print("[Warning] this op param has no output info")

    model_run_params = OpPrams()
    # this is match c++ object, so is camel case
    model_run_params.inputCnt = c_int(len(input_infos))  # pylint: disable=invalid-name, attribute-defined-outside-init
    model_run_params.outputCnt = c_int(  # pylint: disable=invalid-name, attribute-defined-outside-init
        len(output_infos))
    input_data_path = ";".join([x["dataPath"] for x in input_infos])
    output_data_path = ";".join([x["dataPath"] for x in output_infos])
    model_run_params.inputDataPaths = bytes(  # pylint: disable=invalid-name, attribute-defined-outside-init
        input_data_path, encoding="utf8")
    model_run_params.outputDataPaths = bytes(  # pylint: disable=invalid-name, attribute-defined-outside-init
        output_data_path, encoding="utf8")
    model_run_params.binFilePath = bytes(  # pylint: disable=invalid-name, attribute-defined-outside-init
        op_param["binFilePath"], encoding="utf8")
    model_run_params.kernelFuncName = bytes(  # pylint: disable=invalid-name, attribute-defined-outside-init
        op_param["kernelFuncName"], encoding="utf8")
    model_run_params.jsonFilePath = bytes(  # pylint: disable=invalid-name, attribute-defined-outside-init
        op_param["jsonFilePath"], encoding="utf8")
    input_sizes = [calc_input_size(x["shape"], x["dtype"]) for x in input_infos]
    output_sizes = [calc_input_size(x["shape"], x["dtype"]) for x in output_infos]
    input_size_str = ";".join([str(ipt_size) for ipt_size in input_sizes])
    model_run_params.inputSizes = bytes(  # pylint: disable=invalid-name, attribute-defined-outside-init
        input_size_str, encoding="utf8")
    output_size_str = ";".join([str(opt_size) for opt_size in output_sizes])
    model_run_params.outputSizes = bytes(  # pylint: disable=invalid-name, attribute-defined-outside-init
        output_size_str, encoding="utf8")
    return model_run_params


def set_camodel_log_path(ca_model_log_path):
    """
    set_camodel_log_path
        if has "CAMODEL_LOG_PATH" env, and not ca_model_log_path, use default env config
        if has "CAMODEL_LOG_PATH" env, and ca_model_log_path is not none,
            set ca_model_log_path to env, and reset env after call
        if has not "CAMODEL_LOG_PATH" env, and ca_model_log_path is none,
              set "./model" to env, and clear env after call
        if has not "CAMODEL_LOG_PATH" env, and ca_model_log_path is not none,
             set ca_model_log_path to env, and clear env after call
    :param ca_model_log_path: ca model log path
    :return: None
    """

    if ca_model_log_path:
        new_log_path = ca_model_log_path
    old_log_path = os.environ.get(CAMODEL_LOG_PATH_ENV)
    if not ca_model_log_path and not old_log_path:
        new_log_path = "./model"
    if new_log_path:
        new_log_path = os.path.realpath(new_log_path)
        print(new_log_path)
        if not os.path.exists(new_log_path):
            file_util.makedirs(new_log_path, DATA_DIR_MODES)
        os.environ[CAMODEL_LOG_PATH_ENV] = new_log_path

    return new_log_path, old_log_path


def _run_op(params):
    """
    run op on ca model

    :param op_param:  eg:
        op_param = {
            "inputInfos": [{"dtype": "float16", "shape": (1,), "dataPath": "input1.data"},
                           {"dtype": "float16", "shape": (1,), "dataPath": "input1.data"}],
            "outputInfos": [{"dtype": "float16", "shape": (1,), "dataPath": "abc.data"}, ],
            "binFilePath": "add/cce_add_1_1_float32.o",
            "kernelFuncName": "cce_add_1_1_float32__kernel0",
            "jsonFilePath": "kernel_meta/cce_add_1_float32.json"
        }
    :param ca_model_log_path: the ca model dump file saved path, default is ./model
    or your environment config "CAMODEL_LOG_PATH", this param will cover default and your env configuration
    :return:
    """
    global OP_RUN_UTILS  # pylint: disable=global-statement

    op_param = params.get("op_param")
    ca_model_log_path = params.get("ca_model_log_path")

    if not OP_RUN_UTILS:
        load_model_run_tool_so()

    if not check_op_params(op_param):
        raise RuntimeError("Check op params failed")

    model_run_param = build_model_run_params(op_param)

    new_log_path, old_log_path = set_camodel_log_path(ca_model_log_path)

    OP_RUN_UTILS.RunOp(model_run_param)

    if new_log_path:
        if old_log_path:
            os.environ[CAMODEL_LOG_PATH_ENV] = old_log_path
        else:
            del os.environ[CAMODEL_LOG_PATH_ENV]


def _run_op_on_ca(op_param, ca_model_log_path=None):
    with Pool(processes=1) as pool:
        pool.map(_run_op, [{"op_param": op_param, "ca_model_log_path": ca_model_log_path}, ])


def run_op(op_param, ca_model_log_path=None):
    """
    run op on ca model

    :param op_param:  eg:
        op_param = {
            "inputInfos": [{"dtype": "float16", "shape": (1,), "dataPath": "input1.data"},
                           {"dtype": "float16", "shape": (1,), "dataPath": "input1.data"}],
            "outputInfos": [{"dtype": "float16", "shape": (1,), "dataPath": "abc.data"}, ],
            "binFilePath": "add/cce_add_1_1_float32.o",
            "kernelFuncName": "cce_add_1_1_float32__kernel0",
            "jsonFilePath": "kernel_meta/cce_add_1_float32.json"
        }
    :param ca_model_log_path: the ca model dump file saved path, default is:
     ./model or your environment config "CAMODEL_LOG_PATH", this param will cover default and your env configuration
    :return:
    """
    global OP_RUN_UTILS  # pylint: disable=global-statement

    if RUN_MODE == 'ca':
        _run_op_on_ca(op_param, ca_model_log_path)
    else:
        _run_op({"op_param": op_param, "ca_model_log_path": ca_model_log_path})
