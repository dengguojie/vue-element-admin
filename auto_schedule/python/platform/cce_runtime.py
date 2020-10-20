# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
Runtime function related hooks
"""
from __future__ import absolute_import as _abs

import hashlib
import json
import os
import stat

from te.platform import cce_conf
from te.platform import cce_params

from te.tvm.contrib import ccec
from te.tvm.contrib import util
from te.tvm._ffi.function import register_func
from te.tvm.intrin import call_pure_intrin
from te.tvm.intrin import call_extern


# def the pipe line dict
PIPELINES = {10: "PIPE_ALL", 9: "PIPE_V2", 8: 'PIPE_MTE5', 7: 'PIPE_MTE4',
             6: "PIPE_MTE3", 5: "PIPE_MTE2", 4: "PIPE_MTE1", 3: "PIPE_M",
             2: "PIPE_V", 1: "PIPE_S"}
EVENTS = ["EVENT_ID0", "EVENT_ID1", "EVENT_ID2", "EVENT_ID3"]


@register_func("tvm.intrin.rule.cce.cce.coproc_sync")
def coproc_sync(tvm_cce_op):
    """
    coproc sync
    """
    pipe = "PIPE_ALL" if not tvm_cce_op.args else PIPELINES[tvm_cce_op.args[0].value]
    cce_print_params = call_pure_intrin("int32", "tvm_cce_string_print", pipe)
    return call_extern("int32", "pipe_barrier", cce_print_params)


@register_func("tvm.intrin.rule.cce.cce.coproc_dep_push")
def coproc_dep_push(tvm_cce_op):
    """
    coproc deep push
    """
    args = [i.value for i in tvm_cce_op.args]
    cce_print_params = call_pure_intrin("int32", "tvm_cce_string_print",
                                        PIPELINES[args[0]],
                                        PIPELINES[args[1]], EVENTS[args[2]])
    return call_extern("int32", "set_flag", cce_print_params)


@register_func("tvm.intrin.rule.cce.cce.coproc_dep_pop")
def coproc_dep_pop(tvm_cce_op):
    """
    coproc deep pop
    """
    args = [i.value for i in tvm_cce_op.args]
    cce_print_params = call_pure_intrin("int32", "tvm_cce_string_print",
                                        PIPELINES[args[0]],
                                        PIPELINES[args[1]], EVENTS[args[2]])
    return call_extern("int32", "wait_flag", cce_print_params)


# pylint: disable=no-member
@register_func
def tvm_callback_create_tmp_dir():
    """
    create tmp dir
    """
    return util.mk_tmpdir()


# pylint: disable=no-member, too-many-boolean-expressions
def remove_temp_dir(dirpath):
    """
    :param dirpath: the temp_dir path
    :return:
    """
    dirpath = os.path.realpath(dirpath)
    # deal with soft links for real path like:os.path.realpathrealpath('/tmp')
    if dirpath.startswith('/tmp') or dirpath.startswith('/var/tmp')\
            or dirpath.startswith('/usr/tmp') \
            or dirpath.startswith(os.path.realpath('/tmp')) \
            or dirpath.startswith(os.path.realpath('/var/tmp')) \
            or dirpath.startswith(os.path.realpath('/usr/tmp')):
        util.rm_tmpdir(dirpath + "/")


@register_func
def tvm_callback_remove_tmp_dir(dirpath):
    """
    remove tmp dir
    """
    remove_temp_dir(dirpath)


class CceFlag(object):
    """CceFlag"""
    BatchBindOnly = False


@register_func
def tvm_callback_get_cce_output_dir():
    """
    get cce output directory
    """
    path = os.path.join(os.getcwd(), cce_params.OUTPUT_PATH_CLASS.output_path)
    if path[-1] != '/':
        path += '/'
    return path


@register_func
def tvm_callback_mkdir(dir_path):
    """
    create directory if not exist
    """
    if os.path.exists(dir_path) and not os.path.isdir(dir_path):
        raise RuntimeError(
            "The output directory specified should be a directory, "
            "but now is a file")

    if dir_path[-1] != '/':
        dir_path += '/'
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
        except OSError as err:
            raise err


@register_func
def tvm_callback_cce_compile(target, kernel_name, dirpath, json_info=None):
    """
    cce compile
    """
    output_dir = cce_params.OUTPUT_PATH_CLASS.output_path
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
        except OSError as err:
            # 17, OSError: [Errno 17] File exists
            if err.errno == 17:
                pass
            # 2, No such file or directory
            elif err.errno == 2:
                remove_temp_dir(dirpath)
                raise RuntimeError("Invalid directory")
            else:
                remove_temp_dir(dirpath)
                raise err
    if not os.access(output_dir, os.W_OK | os.X_OK):
        raise RuntimeError("directory is not writable", output_dir)

    is_aicpu = False
    if target == "cce_cpu":
        is_aicpu = True
    elif target == "cce_cpu_llvm":
        kernel_name = kernel_name[1:]
        is_aicpu = True

    bin_file_prefix = ""
    bin_file_suffix = ".o"
    cce_product_params = cce_conf.CceProductParams()
    aicpu_support_os = cce_product_params.getParams("Compiler_aicpu_support_os")
    if is_aicpu and aicpu_support_os:
        bin_file_prefix = "lib"
        bin_file_suffix = ".so"

    json_info = json_info or {}
    for key, val in json_info.items():
        if key.name == "batchBindOnly":
            CceFlag.BatchBindOnly = True

    try:
        ccebin = ccec.compile_cce(
            dirpath, kernel_name, target,
            path_target=os.path.join(
                output_dir, bin_file_prefix + kernel_name + bin_file_suffix))
    except Exception as errs:
        remove_temp_dir(dirpath)
        raise RuntimeError("compile cce error : ", errs)

    return ccebin


# pylint: disable=broad-except
def write_code(js_dict, fname):
    """
    write code
    """
    fname = os.path.realpath(fname)
    if fname.startswith(os.getcwd()):
        try:
            with open(fname, "w") as nwe_file:
                # Only the owner and group have rights
                os.chmod(fname, stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP)
                json.dump(js_dict, nwe_file, sort_keys=True, indent=4,
                          separators=(',', ':'))
        except Exception as err:
            raise RuntimeError("open file error, reason:", err)


def add_json_info(title_dict, json_info, json_info_tuple):
    """add json info"""
    json_info = json_info or {}
    json_info_tuple = json_info_tuple or {}
    for key, val in json_info.items():
        title_dict[key.name] = val.value
        if key.name == "batchBindOnly":
            CceFlag.BatchBindOnly = True
    for key, val in json_info_tuple.items():
        list_value = []
        for args in val:
            list_value.append(int(args))
        title_dict[key.name] = list_value
    return title_dict


def _add_kernel_magic(soc_version, aicore_type):
    if soc_version == "Ascend615" and aicore_type == "VectorCore":
        title_dict = {"magic": "RT_DEV_BINARY_MAGIC_ELF_AIVEC"}
    else:
        title_dict = {"magic": "RT_DEV_BINARY_MAGIC_ELF"}
    return title_dict

# pylint: disable=too-many-locals, too-many-branches, too-many-statements
# blockdim: cpu num,default value is 1.
@register_func
def tvm_callback_cce_postproc(target,
                              kernel_name,
                              blockdim=1,
                              args_size=0,
                              atomic_args="",
                              json_info=None,
                              json_info_tuple=None):
    """
    cce postproc
    """
    is_aicpu = False
    if target == "cce_core":
        soc_version = cce_conf.get_soc_spec("SOC_VERSION")
        aicore_type = cce_conf.get_soc_spec("AICORE_TYPE")
        title_dict = _add_kernel_magic(soc_version, aicore_type)
    elif target == "cce_cpu":
        title_dict = {"magic": "RT_DEV_BINARY_MAGIC_ELF_AICPU"}
        is_aicpu = True
    elif target == "cce_cpu_llvm":
        title_dict = {"magic": "RT_DEV_BINARY_MAGIC_ELF_AICPU"}
        if len(kernel_name) < 2:
            raise ValueError("Invalid code, the length of kernel_name in IR code too short")
        kernel_name = kernel_name[1:]
        is_aicpu = True
    else:
        raise ValueError("Unknown architecture, does not support now")

    # get the instance of CcePructParams
    cce_product_params = cce_conf.CceProductParams()
    # for aicpu supports os only
    aicpu_support_os = cce_product_params.getParams("Compiler_aicpu_support_os")

    # bin file without suffix
    bin_file_name = ""
    bin_file_suffix = ".o"
    bin_file_name = kernel_name
    if is_aicpu and aicpu_support_os:
        bin_file_name = "lib" + bin_file_name
        bin_file_suffix = ".so"

    # compute the sha256 of a given tvm_cce_op file used by domi
    sha256_hash = ""
    bin_file_path = os.path.join(cce_params.OUTPUT_PATH_CLASS.output_path,
                                 bin_file_name + bin_file_suffix)
    try:
        with open(bin_file_path, 'rb') as nwe_file:
            # Only the owner and group have rights
            os.chmod(bin_file_path, stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP)
            sha256_hash = hashlib.sha256(nwe_file.read()).hexdigest()
    except IOError:
        raise RuntimeError("Open operator file failed.")

    # dict to write into json
    title_dict["blockDim"] = blockdim
    title_dict["opParaSize"] = args_size
    title_dict = add_json_info(title_dict, json_info, json_info_tuple)

    # new parameters in aicpuos feature
    title_dict["kernelName"] = kernel_name + "__kernel0"
    title_dict["binFileSuffix"] = bin_file_suffix
    title_dict["binFileName"] = bin_file_name
    title_dict["sha256"] = sha256_hash
    # pylint: disable=len-as-condition
    if hasattr(cce_params.TIK_WORKSPACE_SIZE_LIST, "local_list") and \
            len(cce_params.TIK_WORKSPACE_SIZE_LIST.local_list) > 0:
        title_dict["workspace"] = {
            "num": len(cce_params.TIK_WORKSPACE_SIZE_LIST.local_list),
            "size": cce_params. TIK_WORKSPACE_SIZE_LIST.local_list}
    list_value = []
    # pylint: disable=len-as-condition
    for args in atomic_args:
        list_value.append(int(args))
    title_dict["parameters"] = list_value

    if hasattr(cce_params.TIK_ATOMIC_ADD_LIST, "local_list") and \
            len(cce_params.TIK_ATOMIC_ADD_LIST.local_list) > 0:
        title_dict["parameters"] = cce_params.TIK_ATOMIC_ADD_LIST.local_list

    # the tvm_cce_op json file used by domi
    # kernel_name has been checked, it only contains letters, numbers and underscores
    file_name = os.path.join(cce_params.OUTPUT_PATH_CLASS.output_path,
                             kernel_name + ".json")

    if not os.path.exists(cce_params.OUTPUT_PATH_CLASS.output_path):
        try:
            os.mkdir(cce_params.OUTPUT_PATH_CLASS.output_path)
            os.chmod(cce_params.OUTPUT_PATH_CLASS.output_path,
                     stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
        except OSError as err:
            # 17, OSError: [Errno 17] File exists
            if err.errno == 17:
                pass
            else:
                raise err

    final_dict = title_dict.copy()

    # add tvm_cce_op pattern message
    # If get_fuse_info return true, this indicates that this tvm_cce_op is called
    # from the TBE API to prepare for the fusion.
    # If this tvm_cce_op is not called from the TBE API, the field 'pattern' will not
    # be generated.
    from te.platform.fusion_manager import fusion_manager
    if fusion_manager.get_fuse_info():
        final_dict['pattern'] = fusion_manager.get_current_op_pattern()

    write_code(final_dict, file_name)
