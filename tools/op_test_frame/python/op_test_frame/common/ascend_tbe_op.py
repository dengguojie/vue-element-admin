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
ascend_tbe_op module
"""

import os
import sys
import json
import ctypes
from typing import List
from typing import Dict
from typing import Union

import numpy as np

from op_test_frame.runtime import AscendRTSApi
from op_test_frame.common import logger
from op_test_frame.utils import shape_utils

NP_DTYPE_ASCEND_TYPE_MAP = {
    np.float16.__name__: "float16",
    np.float32.__name__: "float32",
    np.float64.__name__: "float64",
    np.bool.__name__: "bool",
    np.int8.__name__: "int8",
    np.uint8.__name__: "uint8",
    np.int16.__name__: "int16",
    np.uint16.__name__: "uint16",
    np.int32.__name__: "int32",
    np.uint32.__name__: "uint32",
    np.int64.__name__: "int64",
    np.uint64.__name__: "uint64"
}

ASCEND_DTYPE_NP_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64
}

DTYPE_SIZE_MAP = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8
}


class AscendOpKernel:
    def __init__(self, bin_path: str, json_path: str):
        if not os.path.exists(bin_path):
            raise IOError("bin_path not exist, path: %s" % bin_path)

        if not os.path.exists(json_path):
            raise IOError("json_path not exist, path: %s" % json_path)

        self.bin_path = bin_path
        self.json_path = json_path
        self._parse_json_file(json_path)
        self.stub_func_p = None
        self.input_infos = []
        self.output_infos = []

    def is_registered_to_device(self):
        return self.stub_func_p is not None

    def set_stub_func_p(self, stub_func_p):
        self.stub_func_p = stub_func_p

    def _parse_json_file(self, json_path):
        with open(json_path) as json_f:
            json_str = json_f.read()

        json_obj = json.loads(json_str)
        self.block_dim = json_obj.get("blockDim")
        self.stub_func_name = json_obj.get("kernelName")
        self.magic = json_obj.get("magic")
        workspace_info = json_obj.get("workspace")
        if not workspace_info:
            self.workspace = []
        else:
            self.workspace = workspace_info.get("size", [])
        op_para_size = json_obj.get("opParaSize", None)
        if not op_para_size:
            self.has_tiling = False
            self.tiling_data_size = 0
        else:
            self.has_tiling = True
            self.tiling_data_size = op_para_size

    def set_input_info(self, input_infos):
        self.input_infos = input_infos

    def set_output_info(self, output_infos):
        self.output_infos = output_infos


def calc_op_param_size(shape_size, dtype):
    if not isinstance(dtype, str) and dtype not in DTYPE_SIZE_MAP.keys():
        raise TypeError("dtype must be str and in [%s]" % ",".join(DTYPE_SIZE_MAP.keys()))
    dtype_size = DTYPE_SIZE_MAP.get(dtype)
    return shape_size * dtype_size


class AscendOp:
    def __init__(self, op_type, op_module_name, op_intf_name):
        if op_type is None or not isinstance(op_type, str):
            raise TypeError("op_type must be a str")
        if op_module_name is None or not isinstance(op_module_name, str):
            raise TypeError("op_module_name must be a str")
        if op_intf_name is None or not isinstance(op_intf_name, str):
            raise TypeError("op_intf_name must be a str")
        self.op_type = op_type
        self.op_module_name = op_module_name
        self.op_intf_name = op_intf_name

    def _load_op_func(self):
        try:
            __import__(self.op_module_name)
        except ImportError as import_err:
            raise RuntimeError(
                "can't import op module, op module name: %s, please check you python path and " % self.op_module_name)

        op_module = sys.modules[self.op_module_name]
        op_func = getattr(op_module, self.op_intf_name)
        if not op_func:
            raise RuntimeError("can't get op function in op module, op module name: %s, op function name: %s"
                               % (self.op_module_name, self.op_intf_name))
        return op_func

    @staticmethod
    def _get_param_type(one_param):
        if not one_param:
            return None
        if isinstance(one_param, (tuple, list)):
            if not one_param or not isinstance(one_param[0], dict):
                return None
            return one_param[0].get("param_type", None)
        if isinstance(one_param, dict):
            return one_param.get("param_type")
        return None

    @staticmethod
    def _get_input_outputs(param_list: List):
        def _add_to_params(params: List, one_param):
            if isinstance(one_param, list):
                for sub_param in one_param:
                    params.append(sub_param)
            else:
                params.append(one_param)

        input_list = []
        output_list = []
        for arg in param_list:
            param_type = AscendOp._get_param_type(arg)
            if param_type == "input":
                _add_to_params(input_list, arg)
            if param_type == "output":
                _add_to_params(output_list, arg)
        return input_list, output_list

    @staticmethod
    def _pick_kernel_args(args):
        input_list, output_list = AscendOp._get_input_outputs(args)
        kernel_input_list = []
        for input_info in input_list:
            shape = input_info.get("shape")
            shape_size = shape_utils.calc_shape_size(shape)
            dtype = input_info.get("dtype")
            size = -1 if shape_size < 0 else calc_op_param_size(shape_size, dtype)
            info = {
                "shape": shape,
                "dtype": dtype,
                "format": input_info.get("format"),
                "size": size
            }
            kernel_input_list.append(info)

        kernel_output_list = []
        for output_info in output_list:
            shape = output_info.get("shape")
            shape_size = shape_utils.calc_shape_size(shape)
            dtype = output_info.get("dtype")
            size = -1 if shape_size < 0 else calc_op_param_size(shape_size, dtype)
            info = {
                "shape": shape,
                "dtype": dtype,
                "format": output_info.get("format"),
                "size": size
            }
            kernel_output_list.append(info)
        return kernel_input_list, kernel_output_list

    def compile(self, *args, **kwargs) -> AscendOpKernel:
        op_func = self._load_op_func()
        try:
            op_func(*args, **kwargs)
        except BaseException as compile_err:
            raise RuntimeError("Compile op failed.") from compile_err

        kernel_name = kwargs.get("kernel_name")
        kernel_meta_dir = os.path.realpath("./kernel_meta")
        bin_path = os.path.join(kernel_meta_dir, kernel_name + ".o")
        json_path = os.path.join(kernel_meta_dir, kernel_name + ".json")
        if not os.path.exists(bin_path) or not os.path.exists(json_path):
            raise RuntimeError("Compile op failed, .o or .json is not generate successful.")

        kernel = AscendOpKernel(bin_path, json_path)

        kernel_inputs, kernel_outputs = self._pick_kernel_args(args)
        kernel.set_input_info(kernel_inputs)
        kernel.set_output_info(kernel_outputs)
        return kernel


class AscendOpKernelParam:
    def __init__(self, np_data=None, shape=None, dtype=None, ascend_device: AscendRTSApi = None,
                 hbm_pointer: ctypes.c_void_p = None):
        if np_data is not None:
            self._np_data = np_data
            self._is_const = True
            self.shape = np_data.shape
            self.dtype = NP_DTYPE_ASCEND_TYPE_MAP.get(np_data.dtype.type.__name__)
        else:
            self._np_data = None
            self._is_const = False
            self.shape = shape
            self.dtype = dtype
        shape_size = shape_utils.calc_shape_size(self.shape)
        if shape_size < 0:
            raise RuntimeError("Shape size < 0.")
        self.size = calc_op_param_size(shape_size, self.dtype)
        self.shape_size = shape_size
        self._hbm_pointer = hbm_pointer
        self._ascend_device = ascend_device

    @staticmethod
    def build_op_param_by_np_data(np_data):
        return AscendOpKernelParam(np_data=np_data)

    @staticmethod
    def build_op_param_by_data_file(data_file_path: str, dtype: str, shape: List[int]):
        if not os.path.exists(data_file_path):
            raise IOError("data_file_path is not exist, path: %s" % data_file_path)
        np_dtype = ASCEND_DTYPE_NP_DTYPE_MAP.get(dtype)
        if not np_dtype:
            raise RuntimeError("dtype must in [%s]" % ",".join(ASCEND_DTYPE_NP_DTYPE_MAP.keys()))
        np_data = np.fromfile(data_file_path, dtype=np_dtype)
        shape_size = shape_utils.calc_shape_size(shape)
        if shape_size < 0:
            raise RuntimeError("Shape size < 0")
        if shape_size > len(np_data):
            raise RuntimeError("Data size(%d) in data_file < shape size(%d)" % (len(np_data), shape_size))
        np_data = np_data[:shape_size].reshape(shape)
        return AscendOpKernelParam(np_data=np_data)

    def sync_from_device(self):
        if self._ascend_device and self._hbm_pointer:
            byte_data, _ = self._ascend_device.get_data_from_hbm(self._hbm_pointer, self.size)
            np_data = np.frombuffer(byte_data, dtype=ASCEND_DTYPE_NP_DTYPE_MAP.get(self.dtype))
            np_data = np_data[:self.shape_size]
            self._np_data = np.reshape(np_data, self.shape)

    def sync_to_device(self, ascend_device: AscendRTSApi):
        self._ascend_device = ascend_device
        self._hbm_pointer = self._ascend_device.copy_bin_to_hbm(self._np_data.tobytes())

    def is_in_device(self):
        return self._hbm_pointer is not None

    def release_device(self):
        if self._ascend_device and self._hbm_pointer:
            self._ascend_device.free(self._hbm_pointer)
            self._hbm_pointer = None

        if self._ascend_device:
            self._ascend_device = None

    def concat_into_kernel_args(self, kernel_args: List):
        kernel_args.append(self._hbm_pointer)

    def get_data(self):
        self.sync_from_device()
        return self._np_data

    def create_ref(self):
        return self


class AscendOpKernelRunner:
    _kernel_params: List[AscendOpKernelParam]

    def __init__(self, simulator_mode=None, device_id=0, soc_version=None, simulator_lib_path=None,
                 simulator_dump_path="./model", auto_copy_device_data=False, profiling=False, profiling_times=1):
        if not isinstance(profiling_times, int):
            raise TypeError("profiling times should be a int.")
        if profiling_times < 1 or profiling_times > 100:
            raise ValueError("profiling times should between [1, 100]")
        self.device_id = device_id
        self.ascend_device = AscendRTSApi(simulator_mode=simulator_mode,
                                          soc_version=soc_version,
                                          simulator_lib_path=simulator_lib_path,
                                          simulator_dump_path=simulator_dump_path)
        self.ascend_device.set_device(device_id=device_id)
        self._stream = self.ascend_device.create_stream()
        self._kernel_params = []
        self.profiling = profiling
        self.profiling_times = profiling_times

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for kernel_param in self._kernel_params:
            kernel_param.release_device()
        self.ascend_device.reset(self.device_id)

    def build_kernel_param(self, data, shape=None, dtype=None) -> AscendOpKernelParam:
        if isinstance(data, str):
            kernel_param = AscendOpKernelParam.build_op_param_by_data_file(data_file_path=data,
                                                                           shape=shape,
                                                                           dtype=dtype)
        else:
            kernel_param = AscendOpKernelParam.build_op_param_by_np_data(np_data=data)
        kernel_param.sync_to_device(self.ascend_device)
        self._kernel_params.append(kernel_param)
        return kernel_param

    def cache_kernel_param(self, param):
        if param not in self._kernel_params:
            self._kernel_params.append(param)

    def _fill_inputs(self, inputs: List[Union[AscendOpKernelParam]], kernel_args: List, input_params: List):
        for input_info in inputs:
            if isinstance(input_info, AscendOpKernelParam):
                if input_info not in self._kernel_params:
                    self._kernel_params.append(input_info)
                if not input_info.is_in_device():
                    input_info.sync_to_device(self.ascend_device)
                input_params.append(input_info)
                input_info.concat_into_kernel_args(kernel_args)
            else:
                input_param = self.build_kernel_param(input_info)
                input_param.concat_into_kernel_args(kernel_args)

    def _fill_workspace(self, kernel: AscendOpKernel, wksp_hbm_pointers: List, kernel_args: List):
        for workspace_size in kernel.workspace:
            wksp_hbm_p = self.ascend_device.malloc(workspace_size + 32)
            wksp_hbm_pointers.append(wksp_hbm_p)
            kernel_args.append(wksp_hbm_p)

    def _fill_outputs(self, kernel: AscendOpKernel,
                      output_input_ref: List[List[int]],
                      actual_output_info: List[Dict],
                      input_params: List[AscendOpKernelParam],
                      output_params: List[AscendOpKernelParam],
                      kernel_args: List):
        output_idx = 0
        output_input_ref_map = dict(output_input_ref) if output_input_ref else {}
        output_info_list = actual_output_info if actual_output_info else kernel.output_infos
        for output_info in output_info_list:
            if output_info is not None:
                if output_idx in output_input_ref_map.keys():
                    output_param = input_params[output_input_ref_map[output_idx]].create_ref()
                else:
                    out_size = output_info.get("size")
                    if not out_size:
                        shape = output_info.get("shape")
                        shape_size = shape_utils.calc_shape_size(shape)
                        dtype = output_info.get("dtype")
                        out_size = -1 if shape_size < 0 else calc_op_param_size(shape_size, dtype)
                    out_hbm_pointer = self.ascend_device.malloc(out_size)
                    output_param = AscendOpKernelParam(shape=output_info.get("shape"),
                                                       dtype=output_info.get("dtype"),
                                                       ascend_device=self.ascend_device,
                                                       hbm_pointer=out_hbm_pointer)
                output_params.append(output_param)
                output_param.concat_into_kernel_args(kernel_args)
                self.cache_kernel_param(output_param)

    def _execute_kernel(self, kernel: AscendOpKernel, kernel_args):
        if self.profiling:
            self.ascend_device.start_online_profiling(self._stream, self.profiling_times)
        if not kernel.is_registered_to_device():
            registered_binary = self.ascend_device.register_device_binary_kernel(kernel.bin_path)
            stub_func_p = self.ascend_device.register_function(registered_binary, kernel.stub_func_name, 0)
            kernel.set_stub_func_p(stub_func_p)

        def _execute_kernel():
            self.ascend_device.launch_kernel(kernel.stub_func_p,
                                             kernel.block_dim,
                                             kernel_args,
                                             len(kernel_args),
                                             None,
                                             self._stream)
            self.ascend_device.synchronize_with_stream(self._stream)

        if self.profiling:
            for i in range(self.profiling_times):
                _execute_kernel()
        else:
            _execute_kernel()

        if self.profiling:
            profiling_data = self.ascend_device.get_online_profiling_data(self._stream, self.profiling_times)
            cost_time = [float(profiling_data[i].totalcycle) for i in range(self.profiling_times)]
            print(cost_time)
            self.ascend_device.start_online_profiling(self._stream)

    def run(self, kernel: AscendOpKernel, inputs, output_input_ref: List[List[int]] = None,
            tiling=None, actual_output_info=None) -> Union[AscendOpKernelParam, List[AscendOpKernelParam], None]:

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        input_params = []
        kernel_args = []
        self._fill_inputs(inputs, kernel_args, input_params)

        output_params = []
        self._fill_outputs(kernel, output_input_ref, actual_output_info, input_params, output_params, kernel_args)
        workspace_hbm_p_list = []
        self._fill_workspace(kernel, workspace_hbm_p_list, kernel_args)
        knl_args = [arg.value for arg in kernel_args]
        self._execute_kernel(kernel, knl_args)
        for workspace_hbm_p in workspace_hbm_p_list:
            self.ascend_device.free(workspace_hbm_p)
        return output_params[0] if len(output_params) == 1 else output_params
