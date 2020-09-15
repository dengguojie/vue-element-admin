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
import inspect
import unittest
import traceback
from enum import Enum
from typing import List
import te.platform.cce_conf as tbe_platform
from op_test_frame.config import llt_config
from op_test_frame.common import logger
from op_test_frame.common import op_status
from op_test_frame.common import precision_info
from op_test_frame.utils import precision_compare_util
from op_test_frame.utils import op_param_util
from op_test_frame.ut import op_ut_case_info
from op_test_frame.model_run_utils import model_run_utils
from op_test_frame.ut import op_ut_runner


class OpUT:
    KERNEL_DIR = "./kernel_meta"

    class OpFuncType(Enum):
        INTF_FUNC = "INTF_FUNC"
        SELECT_FORMAT_FUNC = "SELECT_FORMAT_FUNC"
        CHECK_SUPPORT_TYPE = "CHECK_SUPPORT_TYPE"

    class FuncCache:
        def __init__(self, status, func_type, func, err_msg=None):
            self.status = status
            self.func_type = func_type
            self.func = func
            self.err_msg = err_msg

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
            self.imply_type = "dynamic"
        else:
            self.imply_type = "static"
        if not op_func_name:
            self.op_func_name = _default_lower_op_type()
        else:
            self.op_func_name = op_func_name

        self.select_format_func_name = llt_config.LLTConf.op_select_format_name
        self.check_support_func_name = llt_config.LLTConf.check_support_func_name
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
            except ImportError as e:
                err_msg = "load op failed, can't import op module, op module name: %s, import error msg:%s" % (
                    self.op_module_name, e.args[0])
                logger.log_err(err_msg)
                return None, err_msg

            op_module_inner = sys.modules[self.op_module_name]
            return op_module_inner, None

        def _build_op_func_cache(op_module_inner, err_msg):
            if not op_module_inner:
                op_intf_cache_inner = OpUT.FuncCache(op_status.FAILED, OpUT.OpFuncType.INTF_FUNC, None, err_msg)
                op_select_cache_inner = OpUT.FuncCache(op_status.FAILED, OpUT.OpFuncType.SELECT_FORMAT_FUNC,
                                                       None, err_msg)
                op_check_support_cache_inner = OpUT.FuncCache(op_status.FAILED, OpUT.OpFuncType.CHECK_SUPPORT_TYPE,
                                                              None, err_msg)
            else:
                op_func = getattr(op_module_inner, self.op_func_name)
                if op_func:
                    self.op_func = op_func
                    op_intf_cache_inner = OpUT.FuncCache(op_status.SUCCESS, OpUT.OpFuncType.INTF_FUNC, op_func,
                                                         None)
                else:
                    err_msg = "can't find op func: %s in module: %s" % (self.op_func_name, self.op_module_name)
                    op_intf_cache_inner = OpUT.FuncCache(op_status.FAILED, OpUT.OpFuncType.INTF_FUNC, None,
                                                         err_msg)

                select_format_func = getattr(op_module_inner, llt_config.LLTConf.op_select_format_name, None)
                if select_format_func:
                    op_select_cache_inner = OpUT.FuncCache(op_status.SUCCESS, OpUT.OpFuncType.SELECT_FORMAT_FUNC,
                                                           select_format_func, err_msg)
                else:
                    err_msg = "can't find op func: %s in module: %s" % (
                        llt_config.LLTConf.op_select_format_name,
                        self.op_module_name)
                    op_select_cache_inner = OpUT.FuncCache(op_status.FAILED, OpUT.OpFuncType.SELECT_FORMAT_FUNC,
                                                           None, err_msg)

                check_support_func = getattr(op_module_inner, llt_config.LLTConf.check_support_func_name, None)
                if check_support_func:
                    op_check_support_cache_inner = OpUT.FuncCache(op_status.SUCCESS,
                                                                  OpUT.OpFuncType.CHECK_SUPPORT_TYPE,
                                                                  check_support_func, err_msg)
                else:
                    err_msg = "can't find op func: %s in module: %s" % (
                        llt_config.LLTConf.check_support_func_name,
                        self.op_module_name)
                    op_check_support_cache_inner = OpUT.FuncCache(op_status.FAILED, OpUT.OpFuncType.SELECT_FORMAT_FUNC,
                                                                  None, err_msg)

            return op_intf_cache_inner, op_select_cache_inner, op_check_support_cache_inner

        if soc_version not in self.func_cache:
            self.func_cache[soc_version] = {}
            op_module, err_msg = _load_op_module()
            op_intf_cache, op_select_cache, op_check_support_cache = _build_op_func_cache(op_module, err_msg)
            self.func_cache[soc_version][OpUT.OpFuncType.INTF_FUNC.value] = op_intf_cache
            self.func_cache[soc_version][OpUT.OpFuncType.SELECT_FORMAT_FUNC.value] = op_select_cache
            self.func_cache[soc_version][OpUT.OpFuncType.CHECK_SUPPORT_TYPE.value] = op_check_support_cache

        return self.func_cache[soc_version][func_type.value]

    def _build_test_class(self):
        def get_test_case_set_up_class(op_name):
            @classmethod
            def setUpClass(self):
                # 必须使用@classmethod 装饰器,所有test运行前运行一次
                print("--------------------test %s start-------------------------------" % (op_name))

            return setUpClass

        def get_test_case_tear_down_class(op_name):
            @classmethod
            def tearDownClass(self):
                # 必须使用 @ classmethod装饰器, 所有test运行完后运行一次
                print()
                print("--------------------test %s end-------------------------------" % (op_name))

            return tearDownClass

        op_test_class = type("Test" + self.op_type, (unittest.TestCase,), {})
        setattr(op_test_class, "setUpClass", get_test_case_set_up_class(self.op_type))
        setattr(op_test_class, "tearDownClass", get_test_case_tear_down_class(self.op_type))
        return op_test_class

    def _add_test_op_intf_method(self, case_info: op_ut_case_info.OpUTCase):
        def _get_test_method(op_params):
            def op_test_func(sub_self):
                soc_version = tbe_platform.get_soc_spec("SOC_VERSION")
                ut_case_trace = op_ut_case_info.OpUTCaseTrace(soc_version, case_info)
                self.test_trace_hook.append(ut_case_trace)
                func_info = self._get_op_func_from_cache(soc_version, OpUT.OpFuncType.INTF_FUNC)
                if not func_info.status == op_status.SUCCESS:
                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(status=op_status.FAILED, result=None,
                                                        err_msg=func_info.err_msg))
                    raise AssertionError(func_info.err_msg)

                err_msg = None
                run_op_compile_success = True
                err_trace_str = None
                try:
                    kernel_name_params = {"kernel_name": case_info.case_name}
                    if len(op_params) and 'range' in op_params[0].keys():
                        import te
                        with te.op.dynamic():
                            func_info.func(*op_params, **kernel_name_params)
                    else:
                        func_info.func(*op_params, **kernel_name_params)
                except BaseException as e:
                    if case_info.expect != op_status.SUCCESS:
                        if not isinstance(e, case_info.expect):
                            run_op_compile_success = False
                            err_msg = "Exception class not match actual is: %s, Expect is:%s" % (
                                e.__class__.__name__, case_info.expect.__name__)
                    else:
                        run_op_compile_success = False
                        if len(e.args) == 1:
                            err_msg = e.args[0]
                        else:
                            err_msg = json.dumps(e.args)
                    if not run_op_compile_success:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        trace_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
                        err_trace_str = ""
                        for t_i in trace_info:
                            err_trace_str += t_i

                if not run_op_compile_success:
                    assert_msg = "Run op not as expect, expect: %s, but: %s" % (
                        case_info.expect if case_info.expect else case_info.expect.__name__,
                        op_status.SUCCESS if not run_op_compile_success else op_status.FAILED)
                    if err_msg:
                        assert_msg += ", error msg: %s" % err_msg
                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(status=op_status.FAILED, result=None, err_msg=assert_msg,
                                                        err_trace=err_trace_str))
                    raise AssertionError(assert_msg)
                else:
                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(status=op_status.SUCCESS, result=None, err_msg=None))

            return op_test_func

        setattr(self._test_class, case_info.case_name, _get_test_method(case_info.op_params))

    def add_test_cfg_cov_case(self):
        """
        Use less
        """
        pass

    def _build_op_ut_case_info(self, support_soc, case,
                               case_usage: op_ut_case_info.CaseUsage = op_ut_case_info.CaseUsage.IMPL,
                               case_line_num=None) -> op_ut_case_info.OpUTCase:
        if not "params" in case.keys():
            raise RuntimeError("Not has params info in case")
        case_name = case.get("case_name")
        if not case_name:
            self._auto_gen_case_name_count += 1
            case_name = "test_%s_auto_case_name_%d" % (self.op_type, self._auto_gen_case_name_count)
        # case_name duplicated, auto change name to xxx(1), xxx(2)
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
                                        op_imply_type=self.imply_type)

    def add_case(self, support_soc=None, case=None):
        if not support_soc:
            support_soc = ("all",)

        if not isinstance(support_soc, (tuple, list)):
            support_soc = (support_soc,)

        case_line_num = "unkown"
        for stack in inspect.stack():
            if not stack.filename.endswith("op_ut.py"):
                case_line_num = stack.lineno

        case_info = self._build_op_ut_case_info(support_soc, case, case_line_num=case_line_num)
        self._case_info_map[case_info.case_name] = case_info
        self._add_test_op_intf_method(case_info)

    def _add_precision_test_method(self, case_info: op_ut_case_info.OpUTCase):
        import numpy as np

        def _gen_input_data_path(param_info_inner, param_idx_inner):
            if "data_path" in param_info_inner.keys():
                if case_info.expect_out_fn:
                    param_info_inner["value"] = np.fromfile(param_info_inner["data_path"],
                                                            str(param_info_inner["dtype"]).strip())
                return param_info_inner["data_path"]
            else:
                input_data_path = os.path.join(self.test_data_dir_hook[0], self.op_type)
                if not os.path.exists(input_data_path):
                    os.makedirs(input_data_path)
                cur_param_data_path = os.path.join(input_data_path, case_info.case_name + "_" +
                                                   param_info_inner["param_type"] + str(param_idx_inner) + ".bin")
                if "value" in param_info_inner.keys():
                    param_info_inner["value"].tofile(cur_param_data_path)
                    return cur_param_data_path
                elif "value_range" in param_info_inner.keys():
                    param_value_range_tmp = param_info_inner["value_range"]
                    param_np_buffer = np.random.uniform(param_value_range_tmp[0], param_value_range_tmp[1],
                                                        size=param_info_inner["shape"]).astype(
                        param_info_inner["dtype"])
                    param_np_buffer.tofile(cur_param_data_path)
                    param_info_inner["value"] = param_np_buffer
                    return cur_param_data_path
                else:
                    print("no value range, use default [0.1, 1.0]")
                    param_np_buffer = np.random.uniform(0.1, 1.0, size=param_info_inner["shape"]).astype(
                        param_info_inner["dtype"])
                    param_np_buffer.tofile(cur_param_data_path)
                    param_info_inner["value"] = param_np_buffer
                    return cur_param_data_path

        def _gen_output_data_path(param_info_inner, param_idx_tmp):
            if "data_path" in param_info_inner.keys():
                return param_info_inner["data_path"]
            else:
                return os.path.join(self.test_data_dir_hook[0], self.op_type, case_info.case_name + "_" +
                                    param_info_inner["param_type"] + str(param_idx_tmp) + ".data")

        def _compare_result(actual_tensors, expect_tensors):
            if len(actual_tensors) != len(expect_tensors):
                return False, "actual output count(%d) != expect output count(%d)" % (
                    len(actual_tensors), len(expect_tensors))
            cmp_res_detail = []
            cmp_res_total = True
            for actual_tensor, expect_tensor in zip(actual_tensors, expect_tensors):
                cmp_res = precision_compare_util.compare_precision(expect_tensor, actual_tensor,
                                                                   precision_standard=case_info.precision_standard)
                if not cmp_res.is_success():
                    cmp_res_total = False
                cmp_res_detail.append(cmp_res)

            return cmp_res_total, json.dumps([x.to_json_obj() for x in cmp_res_detail])

        def _gen_expect_result(*calc_func_params_tmp):
            np_op_out_list = case_info.expect_out_fn(*calc_func_params_tmp)
            if not isinstance(np_op_out_list, (tuple, list)):
                np_op_out_list = [np_op_out_list, ]
            np_op_out_list_res = []
            for np_op_out in np_op_out_list:
                np_op_out_list_res.append(np_op_out.reshape([-1]))
            return np_op_out_list_res

        def _get_test_method(op_params):
            def op_test_func(sub_self):
                soc_version = tbe_platform.get_soc_spec("SOC_VERSION")
                ut_case_trace = op_ut_case_info.OpUTCaseTrace(soc_version, case_info)
                self.test_trace_hook.append(ut_case_trace)

                def _run_compile_op():
                    func_info = self._get_op_func_from_cache(soc_version, OpUT.OpFuncType.INTF_FUNC)
                    if not func_info.status == op_status.SUCCESS:
                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(status=op_status.FAILED, result=None,
                                                            err_msg=func_info.err_msg))
                        raise AssertionError("Not found op interface function.")

                    kernel_name_params = {"kernel_name": case_info.case_name}
                    is_error = False
                    err_msg = None
                    err_trace_str = ""
                    try:
                        func_info.func(*op_params, **kernel_name_params)
                    except BaseException as e:
                        is_error = True

                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        trace_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
                        err_trace_str = ""
                        for t_i in trace_info:
                            err_trace_str += t_i

                        if len(e.args) == 1:
                            err_msg = e.args[0]
                        else:
                            err_msg = json.dumps(e.args)
                    if is_error:
                        assert_msg = "Compile op failed, err_msg: %s" % (err_msg)
                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(status=op_status.FAILED, result=None, err_msg=assert_msg,
                                                            err_trace=err_trace_str))
                        raise AssertionError(assert_msg)
                    else:
                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(status=op_status.SUCCESS, result=None, err_msg=None))

                _run_compile_op()

                input_run_infos = []
                output_run_infos = []

                def _build_model_run_param():
                    # test presicion
                    input_idx = 0
                    output_idx = 0
                    for op_param in op_params:
                        if isinstance(op_param, dict) and "param_type" in op_param.keys() and \
                                op_param["param_type"] == "input":
                            data_path = _gen_input_data_path(op_param, input_idx)
                            input_idx += 1
                            input_run_infos.append({
                                "dtype": op_param["dtype"],
                                "shape": op_param["shape"],
                                "dataPath": data_path
                            })
                        if isinstance(op_param, dict) and "param_type" in op_param.keys() and \
                                op_param["param_type"] == "output":
                            data_path = _gen_output_data_path(op_param, output_idx)
                            output_idx += 1
                            output_run_infos.append({
                                "dtype": op_param["dtype"],
                                "shape": op_param["shape"],
                                "dataPath": data_path
                            })
                    op_kernel_bin_path = os.path.join(OpUT.KERNEL_DIR, case_info.case_name + ".o")
                    op_kernel_func_name = case_info.case_name + "__kernel0"
                    op_json_file_path = os.path.join(OpUT.KERNEL_DIR, case_info.case_name + ".json")
                    return {
                        "inputInfos": input_run_infos,
                        "outputInfos": output_run_infos,
                        "binFilePath": op_kernel_bin_path,
                        "kernelFuncName": op_kernel_func_name,
                        "jsonFilePath": op_json_file_path
                    }

                op_run_params = _build_model_run_param()
                # don't support multi case, self.test_model_data_dir_hook[0], self.op_type, case_info.case_name
                model_data_path = os.path.join(os.path.realpath(self.test_model_data_dir_hook[0]),
                                               self._simulator_mode_hook[0], self.op_type, case_info.case_name)

                def _run_on_model():
                    try:
                        model_run_utils.run_op(op_run_params, model_data_path)
                    except BaseException as e:
                        assert_msg = "run op failed(run op on model failed) not as except success. Error msg: %s" % \
                                     e.args[0]
                        ut_case_trace.add_stage_result(
                            op_ut_case_info.OpUTStageResult(status=op_status.FAILED, result=None, err_msg=assert_msg))
                        raise AssertionError(assert_msg)

                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(status=op_status.SUCCESS, result=None, err_msg=None))

                _run_on_model()
                if self._simulator_mode_hook[0] == "tm":
                    return

                def _prepare_cmp_result():
                    actual_result_tensors = []
                    for output_run_info in output_run_infos:
                        actual_tensor = np.fromfile(output_run_info["dataPath"], str(output_run_info["dtype"]).strip())
                        actual_result_tensors.append(actual_tensor)

                    if case_info.expect_out_fn:
                        expect_result_tensors = _gen_expect_result(*case_info.op_params)
                        if not isinstance(expect_result_tensors, (list, tuple)):
                            expect_result_tensors = [expect_result_tensors, ]
                        idx = 0
                        for expect_result_tensor in expect_result_tensors:
                            param_data_path_tmp = os.path.join(self.test_data_dir_hook[0], self.op_type,
                                                               case_info.case_name + "_expect_output" + str(
                                                                   idx) + ".data")
                            expect_result_tensor.tofile(param_data_path_tmp)
                    else:
                        expect_result_tensors = []
                        for op_param in op_params:
                            if isinstance(op_param, dict) and "param_type" in op_param.keys() and \
                                    op_param["param_type"] == "output":
                                if not op_param.get("expect_data_path", None):
                                    cmp_err_msg = "Compare precision failed, has no calc expect out fn, " \
                                                  "and has no data path."
                                    ut_case_trace.add_stage_result(
                                        op_ut_case_info.OpUTStageResult(op_status.FAILED, result=None,
                                                                        err_msg=cmp_err_msg))
                                    raise AssertionError(cmp_err_msg)
                                expect_result_tensors.append(
                                    np.fromfile(op_param["expect_data_path"], op_param["dtype"]))

                    return actual_result_tensors, expect_result_tensors

                actual_out_tensors, expect_out_tensors = _prepare_cmp_result()
                cmp_res, detail_msg = _compare_result(actual_out_tensors, expect_out_tensors)
                if not cmp_res:
                    assert_msg = "run op test failed(compare precision failed) not as except success."
                    assert_msg += " Detail message:" + detail_msg
                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(op_status.FAILED, result=None, err_msg=assert_msg))
                    raise AssertionError(assert_msg)
                else:
                    ut_case_trace.add_stage_result(
                        op_ut_case_info.OpUTStageResult(op_status.SUCCESS, result=None, err_msg=None))

            return op_test_func

        op_test_class = self._test_class
        setattr(op_test_class, case_info.case_name, _get_test_method(case_info.op_params))

    def add_precision_case(self, support_soc=None, case=None):
        if not support_soc:
            support_soc = ("all",)
        if not isinstance(support_soc, (tuple, list)):
            support_soc = (support_soc,)
        case_line_num = "unkown"
        for stack in inspect.stack():
            if not stack.filename.endswith("op_ut.py"):
                case_line_num = stack.lineno

        case_info = self._build_op_ut_case_info(support_soc, case,
                                                op_ut_case_info.CaseUsage.PRECISION,
                                                case_line_num=case_line_num)
        self._case_info_map[case_info.case_name] = case_info
        self._add_precision_test_method(case_info)

    def get_test_case(self, run_soc, case_name=None, case_usage_list: List[op_ut_case_info.CaseUsage] = None) -> \
            List[op_ut_case_info.OpUTSuite]:
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
                                              self.test_model_data_dir_hook,
                                              self._simulator_mode_hook)
            return op_ut

        test_case_list = []
        for one_soc in run_soc:
            test_case_list.append(_get_one_soc_suit(one_soc))
        return test_case_list

    def get_all_test_case_name(self, soc=None):
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
            for soc in soc_list:
                if soc in case_support_soc:
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

    def run(self, soc, case_name=None, simulator_mode=None, simulator_lib_path=None, print_summary=True):
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
        runner = op_ut_runner.OpUTTestRunner(print_summary=print_summary, simulator_mode=simulator_mode,
                                             simulator_lib_path=simulator_lib_path)
        run_rpt = runner.run(op_uts)
        run_rpt.save("ut_report.txt")
        return run_rpt


class BroadcastOpUT(OpUT):
    def __init__(self, op_type, op_module_name=None, op_func_name=None):
        super(BroadcastOpUT, self).__init__(op_type, op_module_name, op_func_name)
        caller = inspect.stack()[1]
        self.case_file = caller.filename

    def add_broadcast_case(self, soc, input_1_info, input_2_info, output_info=None,
                           expect=op_status.SUCCESS, case_name=None):
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

    def add_broadcast_case_simple(self, soc, dtypes, shape1, shape2, expect=op_status.SUCCESS, case_name=None):
        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes,)
        for dtype in dtypes:
            self.add_broadcast_case(soc, (dtype, shape1, "ND"), (dtype, shape2, "ND"), expect=expect,
                                    case_name=case_name)


class ElementwiseOpUT(OpUT):
    def __init__(self, op_type, op_module_name=None, op_func_name=None):
        super(ElementwiseOpUT, self).__init__(op_type, op_module_name, op_func_name)
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

    def add_elewise_case_simple(self, soc, dtypes, shape, expect=op_status.SUCCESS, case_name=None):
        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes,)
        for dtype in dtypes:
            self.add_elewise_case(soc, [dtype, shape, "ND"], expect=expect, case_name=case_name)


class ReduceOpUT(OpUT):
    def __init__(self, op_type, op_module_name=None, op_func_name=None):
        super(ReduceOpUT, self).__init__(op_type, op_module_name, op_func_name)
        caller = inspect.stack()[1]
        self.case_file = caller.filename

    def _build_reduce_op_param(self, input_info, axes, keep_dim=False):
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

    def add_reduce_case(self, soc, input_info, axes, keep_dim=False, expect=op_status.SUCCESS, case_name=None):

        op_params = self._build_reduce_op_param(input_info, axes, keep_dim)
        op_params["expect"] = expect
        op_params["case_name"] = case_name
        self.add_case(soc, op_params)

    def add_reduce_case_simple(self, soc, dtypes, shape, axes, keep_dim=False, expect=op_status.SUCCESS,
                               case_name=None):
        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes,)
        for dtype in dtypes:
            self.add_reduce_case(soc, [dtype, shape, "ND"], axes, keep_dim=keep_dim, expect=expect, case_name=case_name)
