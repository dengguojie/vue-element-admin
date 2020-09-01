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
op ut case info,
apply info classes: UTCaseFileInfo, OpUTSuite, CaseUsage, OpUTCase, OpUTStageResult, OpUTCaseTrace
"""
from enum import Enum
from unittest import TestSuite
from op_test_frame.common import op_status
from op_test_frame.common import precision_info


class UTCaseFileInfo:
    def __init__(self, case_file, op_module_name):
        self.case_file = case_file
        self.op_module_name = op_module_name


class OpUTSuite:
    def __init__(self, soc, soc_suite: TestSuite, test_res_trace_hook, test_data_dir_hook, dump_model_dir_hook,
                 simulator_mode_hook):
        self.soc = soc
        self.soc_suite = soc_suite
        self._test_res_trace = test_res_trace_hook
        self._test_data_dir_hook = test_data_dir_hook
        self._dump_model_dir_hook = dump_model_dir_hook
        self._simulator_mode_hook = simulator_mode_hook

    def clear_test_trace(self):
        self._test_res_trace[:] = []

    def get_test_trace(self):
        return self._test_res_trace

    def set_test_data_dir(self, data_dir):
        self._test_data_dir_hook.insert(0, data_dir)

    def set_dump_model_dir(self, model_dir):
        self._dump_model_dir_hook.insert(0, model_dir)

    def set_simulator_mode(self, simulator_mode):
        self._simulator_mode_hook.insert(0, simulator_mode)


class CaseUsage(Enum):
    IMPL = "compile"
    CHECK_SUPPORT = "check_support"
    SELECT_FORMAT = "op_select_format"
    CFG_COVERAGE_CHECK = "op_config_coverage_check"
    PRECISION = "precision"

    def to_str(self):
        return self.value

    @staticmethod
    def parser_str(type_str):
        str_enum_map = {
            "compile": CaseUsage.IMPL,
            "check_support": CaseUsage.CHECK_SUPPORT,
            "op_select_format": CaseUsage.SELECT_FORMAT,
            "op_config_coverage_check": CaseUsage.CFG_COVERAGE_CHECK,
            "precision": CaseUsage.PRECISION,
        }
        if not type_str or type_str not in str_enum_map.keys():
            return None
        else:
            return str_enum_map[type_str]


class OpUTCase:

    def __init__(self, support_soc=None, op_type=None, case_name=None,
                 op_params=None, expect=None, case_usage: CaseUsage = CaseUsage.IMPL,
                 expect_out_fn=None, case_file=None, case_line_num=None,
                 precision_standard: precision_info.PrecisionStandard = None,
                 op_imply_type="static"):
        self.support_soc = support_soc
        self.op_type = op_type
        self.case_name = case_name
        self.op_params = op_params
        self.expect = expect
        self.case_usage = case_usage
        self.expect_out_fn = expect_out_fn
        self.case_file = case_file
        self.case_line_num = case_line_num
        self.precision_standard = precision_standard
        self.op_imply_type = op_imply_type

    def to_json_obj(self):
        def _build_input_output_obj(info):
            json_obj = {}
            for key, val in info.items():
                if key != "value":
                    json_obj[key] = val
            return json_obj

        return {
            "support_soc": self.support_soc,
            "op_type": self.op_type,
            "case_name": self.case_name,
            "op_params": [op_param if not isinstance(op_param, dict) else _build_input_output_obj(op_param)
                          for op_param in self.op_params],
            "expect": self.expect if isinstance(self.expect, str) else self.expect.__class__.__name__,
            "case_usage": self.case_usage.to_str(),
            "case_file": self.case_file,
            "case_line_num": self.case_line_num,
            "precision_standard": self.precision_standard.to_json_obj() if self.precision_standard else None,
            "op_imply_type": self.op_imply_type
        }

    @staticmethod
    def parser_json_obj(json_obj):
        if not json_obj:
            return None
        return OpUTCase(support_soc=json_obj["support_soc"],
                        op_type=json_obj["op_type"],
                        case_name=json_obj["case_name"],
                        op_params=json_obj["op_params"],
                        expect=json_obj["expect"],
                        case_usage=CaseUsage.parser_str(json_obj["case_usage"]),
                        case_file=json_obj.get("case_file"),
                        case_line_num=json_obj.get("case_line_num"),
                        precision_standard=precision_info.PrecisionStandard.parse_json_obj(
                            json_obj.get("precision_standard")),
                        op_imply_type=json_obj.get("op_imply_type"))


class OpUTStageResult:
    def __init__(self, status, stage_name=None, result=None, err_msg=None, err_trace=None):
        self.status = status
        self.result = result
        self.err_msg = err_msg
        self.err_trace = err_trace
        self.stage_name = stage_name

    def is_success(self):
        return self.status == op_status.SUCCESS

    def to_json_obj(self):
        return {
            "status": self.status,
            "result": self.result,
            "err_msg": self.err_msg,
            "stage_name": self.stage_name,
            "err_trace": self.err_trace
        }

    @staticmethod
    def parser_json_obj(json_obj):
        return OpUTStageResult(json_obj["status"], json_obj["stage_name"], json_obj["result"], json_obj["err_msg"],
                               json_obj["err_trace"])


class OpUTCaseTrace:
    def __init__(self, soc_version, ut_case_info: OpUTCase):
        self.ut_case_info = ut_case_info
        self.stage_result = []
        self.run_soc = soc_version

    def add_stage_result(self, stage_res: OpUTStageResult):
        self.stage_result.append(stage_res)

    def to_json_obj(self):
        return {
            "run_soc": self.run_soc,
            "ut_case_info": self.ut_case_info.to_json_obj(),
            "stage_result": [stage_obj.to_json_obj() for stage_obj in self.stage_result],
        }

    @staticmethod
    def parser_json_obj(json_obj):
        if not json_obj:
            return None
        res = OpUTCaseTrace(json_obj["run_soc"], OpUTCase.parser_json_obj(json_obj["ut_case_info"]))
        res.stage_result = [OpUTStageResult.parser_json_obj(stage_obj) for stage_obj in json_obj["stage_result"]]
        return res
