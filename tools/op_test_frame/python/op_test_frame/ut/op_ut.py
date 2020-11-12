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
the op ut test main class: OpUT, BroadcaseOpUT, ElementwiseOpUT, ReduceOpUT
"""
import os
import sys
import json
import stat
import inspect
import unittest
import traceback
from enum import Enum
from typing import List

import numpy as np

from op_test_frame.config import llt_config
from op_test_frame.common import logger
from op_test_frame.common import op_status
from op_test_frame.common import precision_info
from op_test_frame.utils import shape_utils
from op_test_frame.utils import precision_compare_util
from op_test_frame.utils import op_param_util
from op_test_frame.utils import file_util
from op_test_frame.ut import op_ut_case_info
from op_test_frame.model_run_utils import model_run_utils
from op_test_frame.ut import op_ut_runner


class OpFuncType(Enum):
    """
    Op Func Type Enum, contains INTF_FUNC, SELECT_FORMAT_FUNC, CHECK_SUPPORT_TYPE
    """
    INTF_FUNC = "INTF_FUNC"
    SELECT_FORMAT_FUNC = "SELECT_FORMAT_FUNC"
    CHECK_SUPPORT_TYPE = "CHECK_SUPPORT_TYPE"


class OpImplyType(Enum):
    """
    op imply type Enum, contains STATIC_SHAPE, DYNAMIC_SHAPE
    """
    STATIC_SHAPE = "static_shape"
    DYNAMIC_SHAPE = "dynamic_shape"


class FuncCache:
    """
    op func cache object struct, contains: status, func_type, func, err_msg
    """

    def __init__(self, status, func_type, func, err_msg=None):
        self.status = status
        self.func_type = func_type
        self.func = func
        self.err_msg = err_msg

    def get_status(self):
        """
        :return: self.status
        """
        return self.status

    def get_func_type(self):
        """
        :return: self.func_type
        """
        return self.func_type

    def get_func(self):
        """
        :return: self.func
        """
        return self.func

    def get_err_msg(self):
        """
        :return: self.err_msg
        """
        return self.err_msg


DATA_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
DATA_FILE_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
DATA_DIR_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP


def get_trace_info() -> str:
    """
    get exception trace info
    :return: exception trace info str
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    trace_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
    if not trace_info:
        return "None exception."
    return "".join(trace_info)


class OpUT:  # pylint: disable=too-many-instance-attributes
    """
        OpUT
        example:
        from op_test_frame import OpUT
        ut_case = OpUT("Add", "impl.add", "add")
        ut_case.add_precision_case(...)

        if __name__ == "__main__":
            ut_case.run("Ascend910", simulator_mode="pv", simulator_lib_path="xxx")
    """
    KERNEL_DIR = os.path.realpath("./kernel_meta")
    SOC_VERSION_STR = "SOC_VERSION"

    def __init__(self, op_type, op_module_name=None, op_func_name=None):

        self.op_type = op_type

        def _default_lower_op_type():
            lower_name = ""
            for ele in str(self.op_type):
                if ele.isupper():
                    lower_name += "_" + ele
                else:
                    lower_name += ele
            if lower_name.startswith("_"):
                lower_name = lower_name[1:]
                lower_name = lower_name.lower()
            return lower_name

        if not op_module_name:
            self.op_module_name = "impl." + _default_lower_op_type()
        else:
            self.op_module_name = op_module_name
        if ".dynamic." in self.op_module_name:
            self.imply_type = OpImplyType.DYNAMIC_SHAPE
        else:
            self.imply_type = OpImplyType.STATIC_SHAPE
        if not op_func_name:
            self.op_func_name = _default_lower_op_type()
        else:
            self.op_func_name = op_func_name

        self._select_format_func_name = llt_config.LLTConf.op_select_format_name
        self._check_support_func_name = llt_config.LLTConf.check_support_func_name
        self.func_cache = {}
        self.test_trace_hook = []
        self._test_class = self._build_test_class()
        self._auto_gen_case_name_count = 0
        self._case_name_list = []
        # key: case_name, value: case_info: OpUTCase
        self._case_info_map = {}
        self.test_data_dir_hook = [os.path.realpath("./data")]
        self.test_model_data_dir_hook = [os.path.realpath("./model")]
        self._simulator_mode_hook = ["pv"]

        caller = inspect.stack()[1]
        self.case_file = caller.filename

    def _get_op_func_from_cache(self, soc_version: str, func_type: OpFuncType) -> FuncCache:
        def _load_op_module():
            try:
                __import__(self.op_module_name)
            except ImportError as _:
                load_op_err_trace = get_trace_info()
                load_op_err_msg = "load op failed, can't import op module, op module name: %s, " % self.op_module_name
                load_op_err_msg += "import error trace:%s" % load_op_err_trace
                logger.log_err(load_op_err_msg)
                return None, load_op_err_msg

            op_module_inner = sys.modules[self.op_module_name]
            return op_module_inner, None

        def _get_func_from_op_module(op_module_inner, func_type_inner, func_name_inner):
            op_func = getattr(op_module_inner, func_name_inner, None)
            if op_func:
                func_cache_inner = FuncCache(status=op_status.SUCCESS,
                                             func_type=func_type_inner,
                                             func=op_func,
                                             err_msg=None)
            else:
                ge_op_func_err_msg = "can't find op func: %s in module: %s" % (func_name_inner, self.op_module_name)
                func_cache_inner = FuncCache(status=op_status.FAILED,
                                             func_type=func_type_inner,
                                             func=None,
                                             err_msg=ge_op_func_err_msg)
            return func_cache_inner

        def _build_op_func_cache():
            op_module_inner, load_op_err_msg = _load_op_module()
            if not op_module_inner:
                op_intf_cache_inner = FuncCache(status=op_status.FAILED,
                                                func_type=OpFuncType.INTF_FUNC,
                                                func=None,
                                                err_msg=load_op_err_msg)
                op_select_cache_inner = FuncCache(status=op_status.FAILED,
                                                  func_type=OpFuncType.SELECT_FORMAT_FUNC,
                                                  func=None,
                                                  err_msg=load_op_err_msg)
                op_check_support_cache_inner = FuncCache(status=op_status.FAILED,
                                                         func_type=OpFuncType.CHECK_SUPPORT_TYPE,
                                                         func=None,
                                                         err_msg=load_op_err_msg)
            else:
                op_intf_cache_inner = _get_func_from_op_module(op_module_inner,
                                                               OpFuncType.INTF_FUNC,
                                                               self.op_func_name)
                op_select_cache_inner = _get_func_from_op_module(op_module_inner,
                                                                 OpFuncType.SELECT_FORMAT_FUNC,
                                                                 llt_config.LLTConf.op_select_format_name)
                op_check_support_cache_inner = _get_func_from_op_module(op_module_inner,
                                                                        OpFuncType.CHECK_SUPPORT_TYPE,
                                                                        llt_config.LLTConf.check_support_func_name)

            return op_intf_cache_inner, op_select_cache_inner, op_check_support_cache_inner

        if soc_version not in self.func_cache:
            self.func_cache[soc_version] = {}
            op_intf_cache, op_select_cache, op_check_support_cache = _build_op_func_cache()
            self.func_cache[soc_version][OpFuncType.INTF_FUNC.value] = op_intf_cache
            self.func_cache[soc_version][OpFuncType.SELECT_FORMAT_FUNC.value] = op_select_cache
            self.func_cache[soc_version][OpFuncType.CHECK_SUPPORT_TYPE.value] = op_check_support_cache

        return self.func_cache.get(soc_version, {}).get(func_type.value)

    def _build_test_class(self):
        def _get_test_case_set_up_class(op_name):
            @classmethod
            def set_up_class(arg):
                if arg is not None:
                    print("--------------------test %s start-------------------------------" % op_name)

            return set_up_class

        def _get_test_case_tear_down_class(op_name):
            @classmethod
            def tear_down_class(arg):
                if arg is not None:
                    print()
                    print("--------------------test %s end-------------------------------" % op_name)

            return tear_down_class

        op_test_class = type("Test" + self.op_type, (unittest.TestCase,), {})
        setattr(op_test_class, "setUpClass", _get_test_case_set_up_class(self.op_type))
        setattr(op_test_class, "tearDownClass", _get_test_case_tear_down_class(self.op_type))
        return op_test_class

    def _get_cur_run_soc_version(self):
        try:
            import te.platform.cce_conf as tbe_platform  # pylint: disable=import-outside-toplevel
        except ImportError as _:
            err_trace = get_trace_info()
            return None, err_trace
        return tbe_platform.get_soc_spec(self.SOC_VERSION_STR), None

    def _run_op_compile(self, func_info: FuncCache, case_info: op_ut_case_info.OpUTCase):
        err_msg = None
        run_op_compile_success = True
        err_trace_str = None
        try:
            if not case_info.addition_params:
                kernel_name_params = {"kernel_name": case_info.case_name}
            else:
                case_info.addition_params["kernel_name"] = case_info.case_name
            if self.imply_type == OpImplyType.DYNAMIC_SHAPE:
                import te  # pylint: disable=import-outside-toplevel
                with te.op.dynamic():
                    func_info.func(*case_info.op_params, **kernel_name_params)
            else:
                func_info.func(*case_info.op_params, **kernel_name_params)
        except BaseException as run_err:  # pylint: disable=broad-except
            if case_info.expect != op_status.SUCCESS:
                if case_info.expect != op_status.FAILED and not isinstance(run_err, case_info.expect):
                    run_op_compile_success = False
                    err_msg = "Exception class not match actual is: %s, Expect is:%s" % (
                        run_err.__class__.__name__, case_info.expect.__name__)
            else:
                run_op_compile_success = False
                if len(run_err.args) == 1:
                    err_msg = run_err.args[0]
                else:
                    err_msg = json.dumps(run_err.args)
            if not run_op_compile_success:
                err_trace_str = get_trace_info()
        return run_op_compile_success, err_msg, err_trace_str

    def _add_test_op_intf_method(self, case_info: op_ut_case_info.OpUTCase):

        def _get_test_method():
            def op_test_func(sub_self):  # pylint: disable=unused-argument
                # add ut trace to trace hook, op runner will get trace info from this hook
                soc_version, get_soc_err_msg = self._get_cur_run_soc_version()
                ut_case_trace = op_ut_case_info.OpUTCaseTrace(soc_version, case_info)
                self.test_trace_hook.append(ut_case_trace)

                if soc_version is None:
                    get_soc_failed_stage = op_ut_case_info.OpUTStageResult(status=op_status.FAILED,
                                                                           err_msg="Can not get run soc",
                                                                           err_trace=get_soc_err_msg)
                    ut_case_trace.add_stage_result(get_soc_failed_stage)
                    raise AssertionError("Not found run soc version.")

                def _run_test():
                    # get op interface function
                    func_info = self._get_op_func_from_cache(soc_version, OpFuncType.INTF_FUNC)
                    if not func_info.status == op_status.SUCCESS:
                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(status=op_status.FAILED,
                                                            err_msg=func_info.err_msg))
                        raise AssertionError(func_info.err_msg)
                    # call op intf, compile op
                    run_op_compile_success, err_msg, err_trace_str = self._run_op_compile(func_info, case_info)
                    # add trace stage info
                    if not run_op_compile_success:
                        assert_msg = "Run op not as expect, expect: %s, but: %s" % (
                            case_info.expect if case_info.expect else case_info.expect.__name__,
                            op_status.SUCCESS if not run_op_compile_success else op_status.FAILED)
                        if err_msg:
                            assert_msg += ", error msg: %s" % err_msg
                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(status=op_status.FAILED,
                                                            err_msg=assert_msg,
                                                            err_trace=err_trace_str))
                        raise AssertionError(assert_msg)

                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(status=op_status.SUCCESS))

                try:
                    _run_test()
                except AssertionError as run_err:
                    raise run_err
                except BaseException as run_err:
                    err_trace = get_trace_info()
                    err_msg = "run op compile test failed."
                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(status=op_status.FAILED,
                                                        err_msg=err_msg,
                                                        err_trace=err_trace))
                    raise AssertionError(err_msg) from run_err

            return op_test_func

        setattr(self._test_class, case_info.case_name, _get_test_method())

    def add_test_cfg_cov_case(self, cfg_path_root=None):
        """
        Use less
        """
        if cfg_path_root is None:
            logger.log_err("add_test_cfg_cov_case but cfg_path_root is none.")
            return
        print("Op Type: %s, not support test cfg_cov_case now." % self.op_type)

    def _build_op_ut_case_info(self, support_soc, case,
                               case_usage: op_ut_case_info.CaseUsage = op_ut_case_info.CaseUsage.IMPL,
                               case_line_num=None) -> op_ut_case_info.OpUTCase:
        if "params" not in case.keys():
            raise RuntimeError("Not has params info in case")
        case_name = case.get("case_name")
        if not case_name:
            self._auto_gen_case_name_count += 1
            case_name = "test_%s_auto_case_name_%d" % (self.op_type, self._auto_gen_case_name_count)
        # case_name duplicated, auto change name to xxx__1, xxx__2
        if case_name in self._case_info_map.keys():
            idx = 1
            while idx < 5000:
                tmp_name = case_name + "__%d" % idx
                idx += 1
                if tmp_name not in self._case_info_map.keys():
                    case_name = tmp_name
                    break

        expect = case.get("expect")
        if not expect:
            expect = op_status.SUCCESS

        precision_standard = case.get("precision_standard")
        if precision_standard and not isinstance(precision_standard, precision_info.PrecisionStandard):
            raise RuntimeError("precision_standard is not op_test_frame.common.precision.PrecisionStandard type")

        return op_ut_case_info.OpUTCase(support_soc=support_soc,
                                        op_type=self.op_type,
                                        case_name=case_name,
                                        op_params=case.get("params"),
                                        expect=expect,
                                        case_usage=case_usage,
                                        expect_out_fn=case.get("calc_expect_func"),
                                        case_file=self.case_file,
                                        case_line_num=case_line_num,
                                        precision_standard=precision_standard,
                                        op_imply_type=self.imply_type.value,
                                        addition_params=case.get("addition_params", None))

    def add_case(self, support_soc=None, case=None):
        """
        add a only test op compile case
        :param support_soc: this case can test soc list
        :param case: case info, this is a dict.
        :return: None
        """
        if not support_soc:
            support_soc = ("all",)

        if not isinstance(support_soc, (tuple, list)):
            support_soc = (support_soc,)

        case_line_num = "unknown"
        for stack in inspect.stack():
            if not stack.filename.endswith("op_ut.py"):
                case_line_num = stack.lineno
                break

        case_info = self._build_op_ut_case_info(support_soc, case, case_line_num=case_line_num)
        self._case_info_map[case_info.case_name] = case_info
        self._add_test_op_intf_method(case_info)

    def _run_op_kernel(self, case_info: op_ut_case_info.OpUTCase):
        def _gen_input_data_path(param_info_inner, param_idx_str):
            def _deal_param_data_path():
                input_data_path = param_info_inner.get("data_path")
                if not os.path.exists(input_data_path):
                    raise RuntimeError("Input data path is not exist, path: %s." % input_data_path)
                param_info_inner["data_path"] = os.path.realpath(input_data_path)
                data_type = str(param_info_inner.get("dtype")).strip()
                data_from_file = np.fromfile(param_info_inner.get("data_path"), data_type)
                data_from_file_size = len(data_from_file)
                param_shape = param_info_inner.get("shape")
                param_shape_size = shape_utils.calc_shape_size(param_shape)
                if data_from_file_size < param_shape_size:
                    raise RuntimeError("Input data file size(%s) is len than shape size(%s), dtype is %s. " % (
                        data_from_file_size, param_shape_size, data_type))
                param_info_inner["value"] = data_from_file[:param_shape_size].reshape(param_shape)

            def _deal_no_param_data_path():
                input_data_dir = os.path.join(self.test_data_dir_hook[0], self.op_type)
                if not os.path.exists(input_data_dir):
                    file_util.makedirs(input_data_dir, mode=DATA_DIR_MODES)
                input_data_file_name = "%s_input%s.bin" % (case_info.case_name, param_idx_str)
                input_data_path = os.path.join(input_data_dir, input_data_file_name)
                param_info_inner["data_path"] = input_data_path
                if not os.path.exists(input_data_path):
                    # just for create file
                    with os.fdopen(os.open(input_data_path, DATA_FILE_FLAGS, DATA_FILE_MODES), 'w') as fout:
                        fout.write("")
                if "value" in param_info_inner.keys():
                    param_info_inner["value"].tofile(input_data_path)
                elif "value_range" in param_info_inner.keys():
                    param_value_range_tmp = param_info_inner["value_range"]
                    param_np_buffer = np.random.uniform(param_value_range_tmp[0],
                                                        param_value_range_tmp[1],
                                                        size=param_info_inner["shape"])
                    param_np_buffer = param_np_buffer.astype(param_info_inner["dtype"])
                    param_np_buffer.tofile(input_data_path)
                    param_info_inner["value"] = param_np_buffer
                else:
                    print("no value range, use default [0.1, 1.0]")
                    param_np_buffer = np.random.uniform(0.1, 1.0, size=param_info_inner["shape"])
                    param_np_buffer = param_np_buffer.astype(param_info_inner["dtype"])
                    param_np_buffer.tofile(input_data_path)
                    param_info_inner["value"] = param_np_buffer

            if "data_path" in param_info_inner.keys():
                _deal_param_data_path()
            else:
                _deal_no_param_data_path()

        def _gen_output_data_path(param_info_inner, param_idx_str):
            output_data_path = param_info_inner.get("data_path")
            if not output_data_path:
                output_data_file_name = "%s_output%s.bin" % (case_info.case_name, param_idx_str)
                output_data_dir = os.path.join(self.test_data_dir_hook[0], self.op_type)
                output_data_dir = os.path.realpath(output_data_dir)
                output_data_path = os.path.join(output_data_dir, output_data_file_name)
                if not os.path.exists(output_data_dir):
                    file_util.makedirs(output_data_dir, mode=DATA_DIR_MODES)
                if not os.path.exists(output_data_path):
                    # create output data file with mode
                    with os.fdopen(os.open(output_data_path, DATA_FILE_FLAGS, DATA_FILE_MODES), 'w') as fout:
                        fout.write("")
                param_info_inner["data_path"] = output_data_path
            else:
                output_data_path = os.path.realpath(output_data_path)
                if not os.path.exists(output_data_path):
                    raise RuntimeError("Output data path is not exist, path: %s." % output_data_path)
                param_info_inner["data_path"] = output_data_path

        def _build_model_run_input(one_param, run_input_info_list, input_idx):
            if isinstance(one_param, (tuple, list)):
                sub_idx = 0
                for sub_param in one_param:
                    input_idx_str = "%d_%d" % (input_idx, sub_idx)
                    _gen_input_data_path(sub_param, input_idx_str)
                    sub_idx += 1
                    run_input_info_list.append({
                        "dtype": sub_param.get("dtype"),
                        "shape": sub_param.get("shape"),
                        "dataPath": sub_param.get("data_path")
                    })
            else:
                _gen_input_data_path(one_param, str(input_idx))
                run_input_info_list.append({
                    "dtype": one_param.get("dtype"),
                    "shape": one_param.get("shape"),
                    "dataPath": one_param.get("data_path")
                })

        def _build_model_run_output(one_param, run_output_info_list, output_idx):
            if isinstance(one_param, (tuple, list)):
                sub_idx = 0
                for sub_param in one_param:
                    input_idx_str = "%d_%d" % (output_idx, sub_idx)
                    _gen_output_data_path(sub_param, input_idx_str)
                    sub_idx += 1
                    run_output_info_list.append({
                        "dtype": sub_param.get("dtype"),
                        "shape": sub_param.get("shape"),
                        "dataPath": sub_param.get("data_path")
                    })
            else:
                _gen_output_data_path(one_param, str(output_idx))
                run_output_info_list.append({
                    "dtype": one_param.get("dtype"),
                    "shape": one_param.get("shape"),
                    "dataPath": one_param.get("data_path")
                })

        def _build_model_run_param():
            input_idx = 0
            output_idx = 0
            input_run_infos = []
            output_run_infos = []
            for op_param in case_info.op_params:
                param_type = self._get_param_type(op_param)
                if param_type == "input":
                    _build_model_run_input(op_param, input_run_infos, input_idx)
                    input_idx += 1
                if param_type == "output":
                    _build_model_run_output(op_param, output_run_infos, output_idx)
                    output_idx += 1
            op_kernel_bin_path = os.path.join(OpUT.KERNEL_DIR, case_info.case_name + ".o")
            if not os.path.exists(op_kernel_bin_path):
                raise RuntimeError("Op kernel(.o) is not exist.")
            op_kernel_func_name = case_info.case_name + "__kernel0"
            op_json_file_path = os.path.join(OpUT.KERNEL_DIR, case_info.case_name + ".json")
            if not os.path.exists(op_json_file_path):
                raise RuntimeError("Op kernel(.o) is not exist.")

            return {
                "inputInfos": input_run_infos,
                "outputInfos": output_run_infos,
                "binFilePath": op_kernel_bin_path,
                "kernelFuncName": op_kernel_func_name,
                "jsonFilePath": op_json_file_path
            }

        def _read_output_value(one_param):
            output_data_path = one_param.get("data_path")
            if not os.path.exists(output_data_path):
                raise RuntimeError("Output data path is not exist, path: %s" % output_data_path)
            data_type = one_param.get("dtype")
            if not data_type:
                raise RuntimeError("Output data type in not defined.")
            data_type = data_type.strip()
            data_from_file = np.fromfile(output_data_path, data_type)
            data_from_file_size = len(data_from_file)
            param_shape = one_param.get("shape")
            param_shape_size = shape_utils.calc_shape_size(param_shape)
            if data_from_file_size < param_shape_size:
                raise RuntimeError("Output data file size(%s) is len than shape size(%s), dtype is %s. " % (
                    data_from_file_size, param_shape_size, data_type))
            one_param["value"] = data_from_file[:param_shape_size].reshape(param_shape)

        def _handle_model_run_out():
            for op_param in case_info.op_params:
                param_type = self._get_param_type(op_param)
                if param_type == "output":
                    if isinstance(op_param, (tuple, list)):
                        for sub_param in op_param:
                            _read_output_value(sub_param)
                    else:
                        _read_output_value(op_param)

        # don't support multi case, self.test_model_data_dir_hook[0], self.op_type, case_info.case_name
        model_data_path = os.path.join(os.path.realpath(self.test_model_data_dir_hook[0]),
                                       self._simulator_mode_hook[0], self.op_type, case_info.case_name)
        if not os.path.exists(model_data_path):
            file_util.makedirs(model_data_path, mode=DATA_DIR_MODES)

        try:
            op_run_params = _build_model_run_param()
        except BaseException as _:  # pylint: disable=broad-except
            err_trace = get_trace_info()
            return False, "Build model run param failed", err_trace

        try:
            model_run_utils.run_op(op_run_params, model_data_path)
        except BaseException as _:  # pylint: disable=broad-except
            err_trace = get_trace_info()
            return False, "Run op on model failed", err_trace

        try:
            _handle_model_run_out()
        except BaseException as _:  # pylint: disable=broad-except
            err_trace = get_trace_info()
            return False, "Get model run out data failed.", err_trace

        return True, None, None

    def _run_expect_fn(self, case_info: op_ut_case_info.OpUTCase):

        def _dump_expect_data(one_param, data_tensor):
            if data_tensor is None:
                raise RuntimeError("Expect out tensor is None.")
            one_param["expect_value"] = data_tensor
            output_data_dir = os.path.join(self.test_data_dir_hook[0], self.op_type)
            output_data_dir = os.path.realpath(output_data_dir)
            if not os.path.exists(output_data_dir):
                file_util.makedirs(output_data_dir, mode=DATA_DIR_MODES)
            output_data_file_name = "%s_expect_output%s.bin" % (case_info.case_name, one_param.get("idx_str"))
            output_data_path = os.path.join(output_data_dir, output_data_file_name)
            one_param["expect_data_path"] = output_data_path
            if not os.path.exists(output_data_path):
                # create output data file with mode
                with os.fdopen(os.open(output_data_path, DATA_FILE_FLAGS, DATA_FILE_MODES), 'w') as fout:
                    fout.write("")
            if not getattr(data_tensor, "tofile", None):
                raise RuntimeError("Expect out data is not numpy data, can't use tofile to save data.")

            shape = one_param.get("shape")
            shape_str = ",".join([str(dim) for dim in shape])
            tensor_shape_str = ",".join([str(dim) for dim in data_tensor.shape])
            if shape_str != tensor_shape_str:
                raise RuntimeError("Expect out tensor's shape([%s]) is not match shape([%s]) in op_params." % (
                    tensor_shape_str, shape_str))

            data_tensor.tofile(output_data_path)

        def _get_output_dict():
            output_dict_list = []
            output_idx = 0
            for op_param in case_info.op_params:
                param_type = self._get_param_type(op_param)
                if param_type == "output":
                    if isinstance(op_param, (tuple, list)):
                        sub_idx = 0
                        for sub_param in op_param:
                            output_idx_str = "%d_%d" % (output_idx, sub_idx)
                            sub_param["idx_str"] = output_idx_str
                            output_dict_list.append(sub_param)
                            sub_idx += 1
                    else:
                        output_dict_list.append(op_param)
                        op_param["idx_str"] = str(output_idx)

                    output_idx += 1

            return output_dict_list

        if not case_info.expect_out_fn:
            return False, "Calc expect function is not, please set calc_expect_func when add precision case", None
        addition_params = {}
        if case_info.addition_params:
            addition_params = case_info.addition_params
        try:
            output_tensors = case_info.expect_out_fn(*case_info.op_params, **addition_params)
        except BaseException as _:  # pylint: disable=broad-except
            err_trace = get_trace_info()
            return False, "Run calc_expect_func function failed.", err_trace

        if not isinstance(output_tensors, (list, tuple)):
            output_tensors = [output_tensors, ]
        outputs_dict = _get_output_dict()
        if len(output_tensors) != len(outputs_dict):
            err_msg = "Expect output tensor count(%d), is not match output dict count(%d) in op_params." % (
                len(output_tensors), len(outputs_dict))
            return False, err_msg, None

        try:
            for out_dict, out_tensor in zip(outputs_dict, output_tensors):
                _dump_expect_data(out_dict, out_tensor)
        except BaseException as _:  # pylint: disable=broad-except
            err_trace = get_trace_info()
            return False, "Run dump expect data failed.", err_trace

        return True, None, None

    @staticmethod
    def _get_param_type(one_param):
        if not one_param:
            return None
        if isinstance(one_param, (list, tuple)):
            if not isinstance(one_param[0], dict):
                return None
            return one_param[0].get("param_type")
        if isinstance(one_param, dict):
            return one_param.get("param_type")
        return None

    def _add_precision_test_method(self, case_info: op_ut_case_info.OpUTCase):

        def _get_output_dict():
            output_dict_list = []
            output_idx = 0
            for op_param in case_info.op_params:
                param_type = self._get_param_type(op_param)
                if param_type == "output":
                    if isinstance(op_param, (tuple, list)):
                        sub_idx = 0
                        for sub_param in op_param:
                            output_idx_str = "%d_%d" % (output_idx, sub_idx)
                            sub_param["idx_str"] = output_idx_str
                            output_dict_list.append(sub_param)
                            sub_idx += 1
                    else:
                        output_dict_list.append(op_param)
                        op_param["idx_str"] = str(output_idx)

                    output_idx += 1

            return output_dict_list

        def _compare_result():
            output_dicts = _get_output_dict()
            success = True
            err_msg = ""
            for output_dict in output_dicts:
                cmp_res = precision_compare_util.compare_precision(output_dict.get("value"),
                                                                   output_dict.get("expect_value"),
                                                                   precision_standard=case_info.precision_standard)
                if cmp_res.status != op_status.SUCCESS:
                    success = False
                    err_msg += cmp_res.err_msg
            return success, err_msg

        def _get_test_method():
            def _op_test_func(func_class_self):  # pylint: disable=unused-argument
                soc_version, get_soc_err_msg = self._get_cur_run_soc_version()
                ut_case_trace = op_ut_case_info.OpUTCaseTrace(soc_version, case_info)
                self.test_trace_hook.append(ut_case_trace)
                if soc_version is None:
                    get_soc_failed_stage = op_ut_case_info.OpUTStageResult(status=op_status.FAILED,
                                                                           err_msg="Can not get run soc",
                                                                           err_trace=get_soc_err_msg)
                    ut_case_trace.add_stage_result(get_soc_failed_stage)
                    raise AssertionError("Not found run soc version.")

                def _run_test():
                    def _compile_op():
                        func_info = self._get_op_func_from_cache(soc_version, OpFuncType.INTF_FUNC)
                        if not func_info or func_info.status != op_status.SUCCESS:
                            ut_case_trace.add_stage_result(
                                op_ut_case_info.OpUTStageResult(status=op_status.FAILED, err_msg=func_info.err_msg))
                            raise AssertionError("Not found op interface function.")

                        run_op_compile_success, compile_err_msg, err_trace_str = self._run_op_compile(func_info,
                                                                                                      case_info)
                        if not run_op_compile_success:
                            ut_case_trace.add_stage_result(
                                op_ut_case_info.OpUTStageResult(status=op_status.FAILED, err_msg=compile_err_msg,
                                                                err_trace=err_trace_str))
                            raise AssertionError("Not found op interface function.")

                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(op_status.SUCCESS, result=None, err_msg=None))

                    def _run_op():
                        run_status, err_msg, err_trace_str = self._run_op_kernel(case_info)
                        if not run_status:
                            ut_case_trace.add_stage_result(
                                op_ut_case_info.OpUTStageResult(status=op_status.FAILED, err_msg=err_msg,
                                                                err_trace=err_trace_str))
                            raise AssertionError("Run op on model failed.")

                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(op_status.SUCCESS, result=None, err_msg=None))

                    def _run_expect_out_fn():
                        run_status, err_msg, err_trace_str = self._run_expect_fn(case_info)
                        if not run_status:
                            ut_case_trace.add_stage_result(
                                op_ut_case_info.OpUTStageResult(status=op_status.FAILED, err_msg=err_msg,
                                                                err_trace=err_trace_str))
                            raise AssertionError("Run calc_expect_func function failed.")

                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(op_status.SUCCESS, result=None, err_msg=None))

                    def _cmp_result():
                        cmp_status, err_msg = _compare_result()
                        if not cmp_status:
                            assert_msg_str = "run op test failed(compare precision failed) not as except success."
                            assert_msg_str += " Detail message:" + err_msg
                            ut_case_trace.add_stage_result(
                                op_ut_case_info.OpUTStageResult(op_status.FAILED, result=None, err_msg=assert_msg_str))
                            raise AssertionError(assert_msg_str)

                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(op_status.SUCCESS, result=None, err_msg=None))

                    _compile_op()
                    logger.log_debug("Compile op success")
                    _run_op()
                    logger.log_debug("Run op success")
                    if self._simulator_mode_hook[0] == "tm":
                        return
                    _run_expect_out_fn()
                    logger.log_debug("Calc expect out success")
                    _cmp_result()
                    logger.log_debug("Compare expect out success")

                try:
                    _run_test()
                except AssertionError as run_err:
                    raise run_err
                except BaseException as run_err:  # pylint: disable=broad-except
                    err_trace = get_trace_info()
                    assert_msg = "run precision test failed."
                    assert_msg += " Detail message:" + err_trace
                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(op_status.FAILED, err_msg=assert_msg))
                    raise AssertionError(assert_msg) from run_err

            return _op_test_func

        op_test_class = self._test_class
        setattr(op_test_class, case_info.case_name, _get_test_method())

    def add_precision_case(self, support_soc=None, case=None):
        """
        add a test op compile and precision case
        :param support_soc: support soc list
        :param case: case info
        :return: None
        """
        if not support_soc:
            support_soc = ("all",)
        if not isinstance(support_soc, (tuple, list)):
            support_soc = (support_soc,)
        case_line_num = "unknown"
        for stack in inspect.stack():
            if not stack.filename.endswith("op_ut.py"):
                case_line_num = stack.lineno
                break

        case_info = self._build_op_ut_case_info(support_soc, case,
                                                op_ut_case_info.CaseUsage.PRECISION,
                                                case_line_num=case_line_num)
        self._case_info_map[case_info.case_name] = case_info
        self._add_precision_test_method(case_info)

    def _build_custom_test_case(self, support_soc, test_func, case_line_no):
        case_name = test_func.__name__
        if case_name in self._case_info_map.keys():
            idx = 1
            while idx < 5000:
                tmp_name = case_name + "__%d" % idx
                idx += 1
                if tmp_name not in self._case_info_map.keys():
                    case_name = tmp_name
                    break
        return op_ut_case_info.OpUTCustomCase(support_soc=support_soc,
                                              op_type=self.op_type,
                                              case_name=case_name,
                                              case_usage=op_ut_case_info.CaseUsage.CUSTOM,
                                              case_file=self.case_file,
                                              case_line_num=case_line_no,
                                              test_func_name=test_func.__name__,
                                              test_func=test_func)

    def _add_custom_test_method(self, case_info):
        op_test_class = self._test_class

        def _test_func_inner(test_arg):
            is_call_success = True
            err_trace_str = ""
            err_msg = ""
            soc_version, get_soc_err_msg = self._get_cur_run_soc_version()
            ut_case_trace = op_ut_case_info.OpUTCaseTrace(soc_version, case_info)
            self.test_trace_hook.append(ut_case_trace)

            if soc_version is None:
                get_soc_failed_stage = op_ut_case_info.OpUTStageResult(status=op_status.FAILED,
                                                                       err_msg="Can not get run soc",
                                                                       err_trace=get_soc_err_msg)
                ut_case_trace.add_stage_result(get_soc_failed_stage)
                raise AssertionError("Not found run soc version.")

            try:
                case_info.test_func(test_arg)
            except BaseException as _:  # pylint: disable=broad-except
                err_msg = "run test case failed"
                is_call_success = False
                err_trace_str = get_trace_info()
            if is_call_success:
                ut_case_trace.add_stage_result(
                    op_ut_case_info.OpUTStageResult(status=op_status.SUCCESS, result=None, err_msg=None))
            else:
                ut_case_trace.add_stage_result(
                    op_ut_case_info.OpUTStageResult(status=op_status.FAILED, result=None, err_msg=err_msg,
                                                    err_trace=err_trace_str))

        setattr(op_test_class, case_info.case_name, _test_func_inner)

    def add_cust_test_func(self, support_soc=None, test_func=None):
        """
        add a custom test func
        :param support_soc: support soc version
        :param test_func: should be a func
        :return: None
        """
        if not test_func:
            raise RuntimeError("add_cust_test_func failed, test func is None")
        if not hasattr(test_func, "__call__"):
            raise RuntimeError("add_cust_test_func failed, test func is not a function.")
        if not support_soc:
            support_soc = ("all",)

        if not isinstance(support_soc, (tuple, list)):
            support_soc = (support_soc,)

        case_line_num = "unknown"
        for stack in inspect.stack():
            if not stack.filename.endswith("op_ut.py"):
                case_line_num = stack.lineno
        case_info = self._build_custom_test_case(support_soc, test_func, case_line_num)
        self._case_info_map[case_info.case_name] = case_info
        self._add_custom_test_method(case_info)

    def get_test_case(self, run_soc, case_name=None,
                      case_usage_list: List[op_ut_case_info.CaseUsage] = None) -> List[op_ut_case_info.OpUTSuite]:
        """
        get test case list
        :param run_soc: soc version
        :param case_name: case name, default is none, get all test case
        :param case_usage_list: a CaseUsage list
        :return: return a test suite
        """
        if case_name and not isinstance(case_name, (tuple, list)):
            case_name = [case_name, ]
        if not run_soc:
            logger.log_err("Not set 'run_soc' arg.")
            return []
        if not isinstance(run_soc, (list, tuple)):
            run_soc = (run_soc,)

        def _get_test_method(one_soc_inner):
            one_soc_test_method = []
            found_case = False
            for case_name_tmp, case_info in self._case_info_map.items():
                if case_name:
                    if case_name_tmp not in case_name:
                        continue
                    if case_usage_list and case_info.case_usage not in case_usage_list:
                        logger.log_err(
                            "This case(case_name: %s) case usage not match, need: %s, actual is: %s." % (
                                case_name,
                                ",".join([case_usage_ele.value for case_usage_ele in case_usage_list]),
                                case_info.case_usage.value)
                        )
                    if "all" in case_info.support_soc or one_soc_inner in case_info.support_soc:
                        one_soc_test_method.append(self._test_class(case_name_tmp))
                        found_case = True
                    else:
                        logger.log_err(
                            "This case(case_name: %s) not support this soc(%s)." % (case_name, one_soc_inner))
                else:
                    if ("all" in case_info.support_soc or one_soc_inner in case_info.support_soc) and (
                            not case_usage_list or case_info.case_usage in case_usage_list):
                        one_soc_test_method.append(self._test_class(case_name_tmp))
                        found_case = True

            if not found_case:
                logger.log_warn("Not found any case, op type: %s, run_soc: [%s]" % (self.op_type, one_soc_inner))

            return one_soc_test_method

        def _get_one_soc_suit(one_soc_inner):
            suit = unittest.TestSuite()
            suit.addTests(_get_test_method(one_soc_inner))
            op_ut = op_ut_case_info.OpUTSuite(one_soc_inner, suit, self.test_trace_hook, self.test_data_dir_hook,
                                              self.test_model_data_dir_hook, self._simulator_mode_hook,
                                              op_type=self.op_type)
            return op_ut

        test_case_list = []
        for one_soc in run_soc:
            test_case_list.append(_get_one_soc_suit(one_soc))
        return test_case_list

    def get_all_test_case_name(self, soc=None) -> List:
        """
        get all test case name
        :param soc: soc version, if None get all test case, if not none get support this soc's test case
        :return: test case info list
        """
        soc_list = None
        if isinstance(soc, str):
            if "," in soc:
                soc_list = [soc_str.strip() for soc_str in str(soc).split(",")]
            else:
                soc_list = [soc, ]
        if isinstance(soc, (list, tuple)):
            soc_list = soc

        def _check_soc_match(case_support_soc):
            if not soc_list:
                return True
            if "all" in case_support_soc:
                return True
            for one_soc in soc_list:
                if one_soc in case_support_soc:
                    return True
            return False

        case_info_list = []
        for case_name, case_info in self._case_info_map.items():
            if not _check_soc_match(case_info.support_soc):
                continue
            case_obj = {
                "op_type": case_info.op_type,
                "case_name": case_name,
                "case_usage": case_info.case_usage.value,
                "support_soc": case_info.support_soc
            }
            case_info_list.append(case_obj)
        return case_info_list

    def run(self, soc, case_name=None, simulator_mode=None, simulator_lib_path=None):
        """
        run ut
        :param soc: soc version, one soc or a soc list
        :param case_name: case name, if none will run all test case
        :param simulator_mode: support "pv", "tm"
        :param simulator_lib_path: simulator library path
        :return: None
        """
        if simulator_mode:
            if not simulator_lib_path:
                simulator_lib_path = os.environ.get("SIMULATOR_PATH")
            if not simulator_lib_path:
                raise RuntimeError(
                    "Not configured simulator path, when run simulator. "
                    "Please set simulator_lib_path arg, or set ENV SIMULATOR_PATH")
            if simulator_mode not in model_run_utils.SUPPORT_MODEL_LIST:
                raise RuntimeError("Not support this simulator_mode: %s, not suppot [%s]" % (
                    simulator_mode, ",".join(model_run_utils.SUPPORT_MODEL_LIST)))
            if soc == "all":
                raise RuntimeError("Not support run multi soc in one time when run simulator.")
            self._simulator_mode_hook = [simulator_mode, ]

        op_uts = self.get_test_case(soc, case_name)
        runner = op_ut_runner.OpUTTestRunner(print_summary=True, simulator_mode=simulator_mode,
                                             simulator_lib_path=simulator_lib_path)
        run_rpt = runner.run(op_uts)
        run_rpt.save("ut_report.txt")


class BroadcastOpUT(OpUT):
    """
    OpUT for broadcast op
    """

    def __init__(self, op_type, op_module_name=None, op_func_name=None):
        super().__init__(op_type, op_module_name, op_func_name)
        caller = inspect.stack()[1]
        self.case_file = caller.filename

    def add_broadcast_case(self, soc, input_1_info, input_2_info,  # pylint: disable=too-many-arguments
                           output_info=None, expect=op_status.SUCCESS, case_name=None):
        """
        add a only test op compile case
        :param soc: support soc list
        :param input_1_info: input info, [dtype, shape, format, ori_shape, ori_format]
        :param input_2_info: input info, [dtype, shape, format, ori_shape, ori_format]
        :param output_info: output info, [dtype, shape, format, ori_shape, ori_format]
        :param expect: test case except, default is SUCCESS
        :param case_name: case name, can be none, will auto generate a name
        :return: None
        """
        input_1 = op_param_util.build_op_param(input_1_info)
        input_2 = op_param_util.build_op_param(input_2_info)
        if output_info is None:
            output_param = op_param_util.build_op_param(input_1_info)
            b_shape = op_param_util.broadcast_shape(input_1.get("shape"), input_2.get("shape"))
            output_param["shape"] = b_shape
            b_ori_shape = op_param_util.broadcast_shape(input_1.get("ori_shape"), input_2.get("ori_shape"))
            output_param["ori_shape"] = b_ori_shape
        else:
            output_param = op_param_util.build_op_param(output_info)
        if expect == op_status.SUCCESS:
            self.add_case(soc, {"params": [input_1, input_2, output_param], "case_name": case_name})
        else:
            self.add_case(soc, {"params": [input_1, input_2, output_param], "expect": expect, "case_name": case_name})

    def add_broadcast_case_simple(self, soc, dtypes, shape1, shape2,  # pylint: disable=too-many-arguments
                                  expect=op_status.SUCCESS, case_name=None):
        """
        add a only test op compile case
        :param soc: support soc list
        :param dtypes: need test dtypes
        :param shape1: first input's shape
        :param shape2: second input's shape
        :param expect: test case except, default is SUCCESS
        :param case_name: case name, can be none, will auto generate a name
        :return: None
        """
        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes,)
        for dtype in dtypes:
            self.add_broadcast_case(soc, (dtype, shape1, "ND"), (dtype, shape2, "ND"), expect=expect,
                                    case_name=case_name)


class ElementwiseOpUT(OpUT):
    """
    OpUT for elementwise OpUT
    """

    def __init__(self, op_type, op_module_name=None, op_func_name=None):
        super().__init__(op_type, op_module_name, op_func_name)
        caller = inspect.stack()[1]
        self.case_file = caller.filename

    def add_elewise_case(self, soc, param_info, expect=op_status.SUCCESS, case_name=None):
        """
        :param soc: can be "Ascend910", "Ascend310" ..., and "all" means test all soc
        :param param_info:
                [dtype, shape, format, ori_shape, ori_format] or [dtype, shape, format]
                with 5 element like ["float16", [3,4,5,6], "ND", [3,4,5,6], "ND"]
                with 3 element mean ori_format and ori_shape is the same as format and shape
        :return: None
        """
        input_info = op_param_util.build_op_param(param_info)

        # elementwise op's output is the same as input
        self.add_case(soc, {"params": [input_info, input_info], "expect": expect, "case_name": case_name})

    def add_elewise_case_simple(self, soc, dtypes, shape,  # pylint: disable=too-many-arguments
                                expect=op_status.SUCCESS, case_name=None):
        """
        add a only test op compile case
        :param soc: support soc list
        :param dtypes: need test dtypes
        :param shape: test shape
        :param expect: test case except, default is SUCCESS
        :param case_name: case name, can be none, will auto generate a name
        :return: None
        """
        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes,)
        for dtype in dtypes:
            self.add_elewise_case(soc, [dtype, shape, "ND"], expect=expect, case_name=case_name)


class ReduceOpUT(OpUT):
    """
    OpUT for reduce op
    """

    def __init__(self, op_type, op_module_name=None, op_func_name=None):
        super().__init__(op_type, op_module_name, op_func_name)
        caller = inspect.stack()[1]
        self.case_file = caller.filename

    @staticmethod
    def _build_reduce_op_param(input_info, axes, keep_dim=False):
        input_param = op_param_util.build_op_param(input_info)
        output_shape = input_info[1][:]
        rank = len(output_shape)
        unique_axes = []
        if not isinstance(axes, (tuple, list)):
            axes = [axes, ]
        for axis in axes:
            if axis < -rank or axis >= rank:
                raise RuntimeError("is not in out of rank, shape is [%s], axes is: [%s]" % (
                    ",".join([str(x) for x in output_shape]), ",".join([str(x) for x in axes])))
            if axis < 0:
                axis += rank
            if axis not in unique_axes:
                unique_axes.append(axis)

        reduce_shape = []
        for idx, dim in enumerate(output_shape):
            if idx in unique_axes:
                if keep_dim:
                    reduce_shape.append(1)
            else:
                reduce_shape.append(dim)
        # elementwise op's output is the same as input
        output_info = op_param_util.build_op_param([input_info[0], reduce_shape, input_info[2]])
        return {"params": [input_param, output_info, axes, keep_dim]}

    def add_reduce_case(self, soc, input_info, axes, keep_dim=False,  # pylint: disable=too-many-arguments
                        expect=op_status.SUCCESS, case_name=None):
        """
        add a only test op compile case
        :param soc: support soc list
        :param input_info: input info, [dtype, shape, format, ori_shape, ori_format]
        :param axes: test reduce op's axes's value
        :param keep_dim: test reduce op's attr keep_dim's value
        :param expect: test case except, default is SUCCESS
        :param case_name: case name, can be none, will auto generate a name
        :return: None
        """
        op_params = self._build_reduce_op_param(input_info, axes, keep_dim)
        op_params["expect"] = expect
        op_params["case_name"] = case_name
        self.add_case(soc, op_params)

    def add_reduce_case_simple(self, soc, dtypes, shape, axes, keep_dim=False,  # pylint: disable=too-many-arguments
                               expect=op_status.SUCCESS, case_name=None):
        """
        add a only test op compile case
        :param soc: support soc list
        :param dtypes: need test dtypes
        :param shape: test shape
        :param axes: test reduce op's axes's value
        :param keep_dim: test reduce op's attr keep_dim's value
        :param expect: test case except, default is SUCCESS
        :param case_name: case name, can be none, will auto generate a name
        :return: None
        """
        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes,)
        for dtype in dtypes:
            self.add_reduce_case(soc, [dtype, shape, "ND"], axes, keep_dim=keep_dim, expect=expect, case_name=case_name)
