import os
from enum import Enum
from functools import reduce
from typing import List, Dict
import json
from op_test_frame.common.precision_info import PrecisionStandard
from op_test_frame.common import logger

GEN_CASE_INFO_FILE_NAME = "case_gen_info.json"


class DataDistribute(Enum):
    UNIFORM = "uniform"


class STSingleCaseInfo:
    # def __init__(self, case_name, op, input_desc_list, output_desc_list, attr_list, case_file, case_idx,
    #              parent_case_name=None, gen_case_dir="./out"):
    def __init__(self, case_name, op, input_desc_list, output_desc_list, attr_list, case_file, case_idx,
                 parent_case_name=None):
        self.op = op
        self.case_name = str(case_name).lower().replace(".", "_")
        self.case_file = case_file
        self.case_idx = case_idx
        self.parent_case_name = parent_case_name
        # input, output, attr info
        self.input_desc_list = input_desc_list
        self.output_desc_list = output_desc_list
        self.attr_list = attr_list

        self.np_op = None

        # self.om_file_path = None
        self.input_data_file_paths = []
        self.expect_out_data_file_paths = []
        self.ame_out_data_file_dir = os.path.join(str(self.op).lower(), "data", self.case_name + "_output_tmp_dir")

        self.atc_op_json_file_name = self.case_name + ".json"
        self.atc_om_root_path = os.path.join(str(self.op).lower(), "om")
        self.atc_om_file_name = self.case_name + ".om"

        self.case_run_py_path = os.path.join(str(self.op).lower(), "run.py")
        self.case_init_py_path = os.path.join(str(self.op).lower(), "__init__.py")
        self._auto_gen_input_data_file_path()
        # self.om_file = None
        # self.relate_om_file_path = None
        #
        # self.input_data_files = []
        # self.relate_input_paths = []
        # self.output_data_files = []
        # self.relate_output_paths = []
        # all_case_root = os.path.realpath(gen_case_dir)
        # self.case_gen_root = os.path.join(all_case_root, self.op.lower())
        # self.relate_output_tmp_dir = os.path.join("data", self.case_name + "_output_tmp_dir")

    # def gen_relate_path_for_run(self):
    #     root_len = len(self.case_gen_root) + 1
    #     self.relate_om_file_path = self.om_file[root_len:]
    #     for input_path in self.input_data_files:
    #         self.relate_input_paths.append(input_path[root_len:])
    #     for output_path in self.output_data_files:
    #         self.relate_output_paths.append(output_path[root_len:])

    def _auto_gen_input_data_file_path(self):
        for idx, input_desc in enumerate(self.input_desc_list):
            # may be an option input
            if not input_desc or "type" not in input_desc.keys():
                continue
            else:
                self.input_data_file_paths.append(
                    os.path.join(str(self.op).lower(), "data",
                                 self.case_name + "_input" + str(idx) + "_" + input_desc["type"] + ".bin")
                )

    def gen_np_out_data_path(self):
        self.expect_out_data_file_paths = []
        for idx, output_desc in enumerate(self.output_desc_list):
            self.expect_out_data_file_paths.append(
                os.path.join(str(self.op).lower(), "data",
                             self.case_name + "_expect_np_output" + str(idx) + "_" + output_desc["type"] + ".bin")
            )

    def get_np_out_data_paths(self):
        return self.expect_out_data_file_paths

    def get_atc_json_obj(self):
        return {
            "op": self.op,
            "input_desc": [{
                "format": input_desc["format"],
                "shape": input_desc["shape"],
                "type": input_desc["type"]
            } for input_desc in self.input_desc_list],
            "output_desc": [{
                "format": output_desc["format"],
                "shape": output_desc["shape"],
                "type": output_desc["type"]
            } for output_desc in self.output_desc_list],
            "attr": self.attr_list
        }


class STMultiCaseInfo:
    # def __init__(self, case_name, op, input_desc_list, output_desc_list, attr_list, case_file, case_idx,
    #              gen_case_dir="./out"):
    def __init__(self, case_name, op, input_desc_list, output_desc_list, attr_list, case_file, case_idx):
        self.op = op
        self.case_name = case_name
        # mean case index in case file, for find case file
        self.case_file = case_file
        # mean case index in case file, for find case position
        self.case_idx = case_idx

        self.input_desc_list = input_desc_list
        self.output_desc_list = output_desc_list
        self.attr_list = attr_list

        self.single_case_list = None
        # self.gen_case_dir = gen_case_dir

    def _cross_inputs(self):
        def _get_repeat_num(desc, cross_keys):
            cnt = 1
            for key in cross_keys:
                key_cnt = len(desc[key])
                if key_cnt != 1:
                    if cnt == 1:
                        cnt = key_cnt
                    else:
                        if key_cnt != cnt:
                            raise RuntimeError(
                                "can't crosse gen single case, count is different, case file: %s, case idx: %d" % (
                                    self.case_file, self.case_idx))
            return cnt

        for input_desc in self.input_desc_list:
            if "cross_list" not in input_desc.keys():
                input_desc["cross_list"] = [["format", ], ["shape", ], ["type", ], ["value_range", ],
                                            ["data_distribute", ]]
            if "value_range" not in input_desc.keys():
                input_desc["value_range"] = [[0.1, 1.0], ]
            if "data_distribute" not in input_desc.keys():
                input_desc["data_distribute"] = [DataDistribute.UNIFORM.value, ]
            if not isinstance(input_desc["format"], (tuple, list)):
                input_desc["format"] = [input_desc["format"], ]
            if not isinstance(input_desc["type"], (tuple, list)):
                input_desc["type"] = [input_desc["type"], ]
            if not isinstance(input_desc["value_range"][0], (tuple, list)):
                input_desc["value_range"] = [input_desc["value_range"], ]
            if not isinstance(input_desc["data_distribute"], (tuple, list)):
                input_desc["data_distribute"] = [input_desc["data_distribute"], ]
            group_cross_cnt = []
            for cross_group in input_desc["cross_list"]:
                dup_cnt = _get_repeat_num(input_desc, cross_group)
                group_cross_cnt.append(dup_cnt)
                for key in cross_group:
                    if len(input_desc[key]) != dup_cnt:
                        input_desc[key] = input_desc[key] * dup_cnt
            if len(group_cross_cnt) > 1:
                for i, val in enumerate(group_cross_cnt):
                    if i == 0:
                        continue
                    for pre_idx in range(i):
                        for key in input_desc["cross_list"][pre_idx]:
                            input_desc[key] = input_desc[key] * val
                    repeat_cnt = reduce(lambda x, y: x * y, group_cross_cnt[:i])
                    for key in input_desc["cross_list"][i]:
                        input_desc[key] = reduce(lambda x, y: x + y, [[ele, ] * repeat_cnt for ele in input_desc[key]])
            total_cnt = reduce(lambda x, y: x * y, group_cross_cnt)
            input_desc["total_cnt"] = total_cnt

    def _cross_outputs(self):
        def _get_repeat_num(desc, cross_keys):
            cnt = 1
            for key in cross_keys:
                key_cnt = len(desc[key])
                if key_cnt != 1:
                    if cnt == 1:
                        cnt = key_cnt
                    else:
                        if key_cnt != cnt:
                            raise RuntimeError(
                                "can't crosse gen single case, count is different, case file: %s, case idx: %d" % (
                                    self.case_file, self.case_idx))
            return cnt

        input_dtype_cnt = 1 if not self.input_desc_list else len(self.input_desc_list[0]["type"])

        for output_desc in self.output_desc_list:
            if "cross_list" not in output_desc.keys():
                output_desc["cross_list"] = [["format", ], ["shape", ], ["type", ]]
            if not isinstance(output_desc["format"], (tuple, list)):
                output_desc["format"] = [output_desc["format"], ]
            if not isinstance(output_desc["type"], (tuple, list)):
                output_desc["type"] = [output_desc["type"], ]
            group_cross_cnt = []
            for cross_group in output_desc["cross_list"]:
                dup_cnt = _get_repeat_num(output_desc, cross_group)
                group_cross_cnt.append(dup_cnt)
                for key in cross_group:
                    if len(output_desc[key]) != dup_cnt:
                        output_desc[key] = output_desc[key] * dup_cnt
            if len(group_cross_cnt) > 1:
                for i, val in enumerate(group_cross_cnt):
                    if i == 0:
                        continue
                    for pre_idx in range(i):
                        for key in output_desc["cross_list"][pre_idx]:
                            output_desc[key] = output_desc[key] * val
                    repeat_cnt = reduce(lambda x, y: x * y, group_cross_cnt[:i])
                    for key in output_desc["cross_list"][i]:
                        output_desc[key] = reduce(lambda x, y: x + y,
                                                  [[ele, ] * repeat_cnt for ele in output_desc[key]])

            total_cnt = reduce(lambda x, y: x * y, group_cross_cnt)
            for key in ["format", "shape", "type"]:
                repeat_cnt_by_input = input_dtype_cnt // len(output_desc[key])
                repeat_cnt_by_input = max(1, repeat_cnt_by_input)
                output_desc[key] = output_desc[key] * repeat_cnt_by_input

            output_desc["total_cnt"] = total_cnt

    def gen_single_cases(self):
        if self.single_case_list:
            return self.single_case_list
        self._cross_inputs()
        self._cross_outputs()
        single_cnt = 1 if not self.input_desc_list else len(self.input_desc_list[0]["type"])
        single_case_list = []
        for i in range(single_cnt):
            single_case = STSingleCaseInfo(
                self.case_name + "_subcase_" + str(i),
                self.op,
                [{
                    "format": input_desc["format"][i],
                    "shape": input_desc["shape"][i],
                    "type": input_desc["type"][i],
                    "value_range": input_desc["value_range"][i],
                    "data_distribute": input_desc["data_distribute"][i],
                } for input_desc in self.input_desc_list],
                [{
                    "format": output_desc["format"][i],
                    "shape": output_desc["shape"][i],
                    "type": output_desc["type"][i]
                } for output_desc in self.output_desc_list],
                self.attr_list,
                self.case_file,
                self.case_idx,
                self.case_name,
            )
            single_case_list.append(single_case)
        self.single_case_list = single_case_list
        return single_case_list


class STCasePrecisionReport:
    SUCCESS = "success"
    FAILED = "failed"

    def __init__(self, soc_version, case_name, status, err_msg=None):
        self.soc_version = soc_version
        self.case_name = case_name
        self.status = status
        self.err_msg = err_msg

    def is_success(self):
        if self.status == STCasePrecisionReport.SUCCESS:
            return True
        return False

    def to_json_obj(self):
        return {
            "soc_version": self.soc_version,
            "case_name": self.case_name,
            "status": self.status,
            "err_msg": self.err_msg
        }

    @staticmethod
    def parse_json_obj(json_obj):
        if not json_obj:
            return None
        else:
            return STCasePrecisionReport(json_obj['soc_version'], json_obj['case_name'], json_obj['status'],
                                         json_obj['err_msg'])


class STGenCaseInfo:
    def __init__(self, soc_version, op, case_name, om_file, input_files, output_files, expect_outfiles,
                 precision_standard: PrecisionStandard):
        self.soc_version = soc_version
        self.op = op
        self.case_name = case_name
        self.om_file = om_file
        self.input_files = input_files
        self.output_files = output_files
        self.expect_outfiles = expect_outfiles
        self.precision_standard = precision_standard

    def to_json_obj(self):
        return {
            "soc_version": self.soc_version,
            "op": self.op,
            "case_name": self.case_name,
            "om_file": self.om_file,
            "input_files": self.input_files,
            "output_files": self.output_files,
            "expect_outfiles": self.expect_outfiles,
            "precision_standard": None if not self.precision_standard else self.precision_standard.to_json_obj()
        }

    @staticmethod
    def parse_json_obj(json_obj):
        return STGenCaseInfo(json_obj["soc_version"], json_obj["op"], json_obj["case_name"], json_obj["om_file"],
                             json_obj["input_files"], json_obj["output_files"], json_obj["expect_outfiles"],
                             PrecisionStandard.parse_json_obj(json_obj["precision_standard"]))


def dump_gen_case_info(case_dir: str, gen_case_infos: Dict[str, Dict[str, STGenCaseInfo]]):
    case_info_file_full_path = os.path.join(case_dir, GEN_CASE_INFO_FILE_NAME)
    json_obj = {}
    for soc, case_list in gen_case_infos.items():
        if soc not in json_obj.keys():
            json_obj[soc] = {}
        for case_name, case_info in case_list.items():
            json_obj[soc][case_name] = case_info.to_json_obj()
    json_str = json.dumps(json_obj, indent=4)
    with open(case_info_file_full_path, 'w+') as c_f:
        c_f.write(json_str)


def get_gen_case_info(case_dir: str) -> Dict[str, Dict[str, STGenCaseInfo]]:
    case_info_file_full_path = os.path.join(case_dir, GEN_CASE_INFO_FILE_NAME)
    if not os.path.exists(case_info_file_full_path):
        logger.log_warn("Case gen info file not exist, file path: %s" % case_info_file_full_path)
        return {}
    with open(case_info_file_full_path) as c_f:
        json_str = c_f.read()
    json_obj = json.loads(json_str)
    result = {}
    for soc, soc_cases in json_obj.items():
        if soc not in result.keys():
            result[soc] = {}
        for case_name, case in soc_cases.items():
            result[soc][case_name] = STGenCaseInfo.parse_json_obj(case)
        logger.log_info("Soc: %s, Case count under this soc: %d, in case gen info file." % (soc, len(soc_cases)))
    return result
