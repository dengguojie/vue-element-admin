import json
import os
import shutil
import time
from typing import List

CASE_GEN_REPORT_FILE_NAME = "case_gen_report.json"
PRECISION_REPORT_FILE_NAME = "precision_report.json"
MODEL_RUN_REPORT_FILE_NAME = "run_report.json"
ST_GEN_REPORT_FILE_NAME = "st_gen_report.json"
ST_RUN_REPORT_FILE_NAME = "st_report.py"


class STCommonReport:
    SUCCESS = "success"
    FAILED = "failed"

    def __init__(self, soc_version, op, case_name, status, err_msg=None):
        self.soc_version = soc_version
        self.op = op
        self.case_name = case_name
        self.status = status
        self.err_msg = err_msg

    def is_success(self):
        if self.status == STCommonReport.SUCCESS:
            return True
        return False

    def to_json_obj(self):
        return {
            "soc_version": self.soc_version,
            "op": self.op,
            "case_name": self.case_name,
            "status": self.status,
            "err_msg": self.err_msg
        }


class STCaseModelRunReport(STCommonReport):

    def __init__(self, soc_version, op, case_name, status, err_msg=None):
        super(STCaseModelRunReport, self).__init__(soc_version, op, case_name, status, err_msg)

    def to_json_obj(self):
        return super(STCaseModelRunReport, self).to_json_obj()

    @staticmethod
    def parse_json_obj(json_obj):
        return STCaseModelRunReport(json_obj['soc_version'], json_obj['op'], json_obj['case_name'], json_obj['status'],
                                    json_obj['err_msg'])


class STCasePrecisionReport(STCommonReport):

    def __init__(self, soc_version, op, case_name, status, err_msg=None):
        super(STCasePrecisionReport, self).__init__(soc_version, op, case_name, status, err_msg)

    def to_json_obj(self):
        return super(STCasePrecisionReport, self).to_json_obj()

    @staticmethod
    def parse_json_obj(json_obj):
        return STCasePrecisionReport(json_obj['soc_version'], json_obj['op'], json_obj['case_name'], json_obj['status'],
                                     json_obj['err_msg'])


class STCaseModelGenReport(STCommonReport):

    def __init__(self, soc_version, op, case_name, status, err_msg=None, json_file_path=None, om_file_path=None):
        super(STCaseModelGenReport, self).__init__(soc_version, op, case_name, status, err_msg)
        self.json_file_path = json_file_path
        self.om_file_path = om_file_path

    def is_success(self):
        if self.status == STCasePrecisionReport.SUCCESS:
            return True
        return False


class STCaseDataGenReport:
    SUCCESS = "success"
    FAILED = "failed"

    def __init__(self, case_name, need_gen_expect_output=False):
        self.case_name = case_name
        self.status = STCaseDataGenReport.SUCCESS
        self.err_msg = ""
        self.input_data_file_list = []
        self.input_data_list = []
        self.need_gen_expect_out = need_gen_expect_output
        self.expect_out_data_file_list = []
        self.actual_out_data_file_list = []

    def set_status(self, status, err_msg):
        self.status = status
        self.err_msg = err_msg

    def set_expect_out_info(self, expect_out_data_file_list, actual_out_data_file_list):
        self.expect_out_data_file_list = expect_out_data_file_list
        self.actual_out_data_file_list = actual_out_data_file_list

    def set_input_data_info(self, input_data_file_list, input_data_list):
        self.input_data_file_list = input_data_file_list
        self.input_data_list = input_data_list

    def is_success(self):
        if self.status == STCasePrecisionReport.SUCCESS:
            return True
        return False


class STCaseGenReport(STCommonReport):
    NOT_NEED = "NotNeed"

    def __init__(self, soc_version, op, case_name, status, err_msg=None):
        super(STCaseGenReport, self).__init__(soc_version, op, case_name, status, err_msg)
        self.input_data_gen_status = None
        self.expect_out_gen_status = None
        self.model_gen_status = None

    def refresh_model_gen_rpt(self, rpt: STCaseModelGenReport):
        self.model_gen_status = rpt.status
        if not rpt.is_success():
            if not self.err_msg:
                self.err_msg = ""
            self.err_msg += rpt.err_msg + " "

    def refresh_data_gen_rpt(self, rpt: STCaseDataGenReport):
        self.model_gen_status = rpt.status
        if not rpt.need_gen_expect_out:
            self.expect_out_gen_status = STCaseGenReport.NOT_NEED
        else:
            self.expect_out_gen_status = STCaseGenReport.FAILED
        self.input_data_gen_status = rpt.status
        if self.is_success() and not rpt.is_success():
            self.status = STCaseGenReport.FAILED

    def to_json_obj(self):
        return {
            "soc_version": self.soc_version,
            "op": self.op,
            "case_name": self.case_name,
            "status": self.status,
            "err_msg": self.err_msg,
            "input_data_gen_status": self.input_data_gen_status,
            "expect_out_gen_status": self.expect_out_gen_status,
            "model_gen_status": self.model_gen_status
        }

    @staticmethod
    def parse_json_obj(json_obj):
        rpt = STCaseGenReport(json_obj['soc_version'], json_obj['op'], json_obj['case_name'], json_obj['status'],
                              json_obj['err_msg'])
        rpt.input_data_gen_status = json_obj['input_data_gen_status']
        rpt.expect_out_gen_status = json_obj['expect_out_gen_status']
        rpt.model_gen_status = json_obj['model_gen_status']
        return rpt


class STCaseReport(STCommonReport):
    def __init__(self, soc_version, op, case_name, case_file_name, status, err_msg):
        super(STCaseReport, self).__init__(soc_version, op, case_name, status, err_msg)
        self.case_type = "FUNC"
        self.case_file_name = case_file_name
        self.case_single_file = None

        self.gen_case_success = None
        self.run_case_success = None
        self.precision_success = None
        # 精度分布，最大精度误差，超限精度误差比例
        self.precision_detail = None
        self.precision_standard = {
            "precision_abs_ratio": 0.01,
            "precision_per_ratio": 0.01
        }

        self.performance_success = None
        self.performance_detail = None
        self.performance_gold = None
        self.performance_ratio = None

    def refresh_case_gen_rpt(self, rpt: STCaseGenReport):
        self.gen_case_success = rpt.status
        if self.is_success() and not rpt.is_success():
            self.status = STCaseReport.FAILED
        if not rpt.is_success():
            self.err_msg += rpt.err_msg

    def refresh_case_run_rpt(self, rpt: STCaseModelRunReport):
        self.run_case_success = rpt.status
        if self.is_success() and not rpt.is_success():
            self.status = STCaseReport.FAILED
        if not rpt.is_success():
            self.err_msg += rpt.err_msg

    def refresh_case_precision_rpt(self, rpt: STCasePrecisionReport):
        self.precision_success = rpt.status
        if self.is_success() and not rpt.is_success():
            self.status = STCaseReport.FAILED
        if not rpt.is_success():
            self.err_msg += rpt.err_msg

    def get_generate_txt_report(self):
        return "[%s] Op Type: %s, Case Name: %s, Status Detail: Generate: %s, Model Run: %s, Precision Compare:%s," \
               " Case File: %s" % ("Unkown" if not self.status else self.status,
                                   self.op, self.case_name,
                                   "Unkown" if not self.gen_case_success else self.gen_case_success,
                                   "Unkown" if not self.run_case_success else self.run_case_success,
                                   "Unkown" if not self.precision_success else self.precision_success,
                                   self.case_file_name)

    def to_json_obj(self):
        json_obj = super(STCaseReport, self).to_json_obj()

        json_obj["case_type"] = self.case_type
        json_obj["case_file_name"] = self.case_file_name
        json_obj["case_single_file"] = self.case_single_file
        json_obj["gen_case_success"] = self.gen_case_success
        json_obj["run_case_success"] = self.run_case_success
        json_obj["precision_success"] = self.precision_success
        json_obj["precision_detail"] = self.precision_detail
        json_obj["performance_success"] = self.performance_success
        json_obj["performance_detail"] = self.performance_detail
        json_obj["performance_gold"] = self.performance_gold
        json_obj["performance_ratio"] = self.performance_ratio

        return json_obj

    @staticmethod
    def parser_json_obj(json_obj):
        case_rpt = STCaseReport(json_obj["soc_version"], json_obj["op"], json_obj["case_name"],
                                json_obj["case_file_name"], json_obj["status"], json_obj["err_msg"])
        case_rpt.case_type = json_obj["case_type"]
        case_rpt.case_file_name = json_obj["case_file_name"]
        case_rpt.case_single_file = json_obj["case_single_file"]
        case_rpt.gen_case_success = json_obj["gen_case_success"]
        case_rpt.run_case_success = json_obj["run_case_success"]
        case_rpt.precision_success = json_obj["precision_success"]
        case_rpt.precision_detail = json_obj["precision_detail"]
        case_rpt.performance_success = json_obj["performance_success"]
        case_rpt.performance_detail = json_obj["performance_detail"]
        case_rpt.performance_gold = json_obj["performance_gold"]
        case_rpt.performance_ratio = json_obj["performance_ratio"]
        return case_rpt


class STReport:
    def __init__(self):
        self.case_list = []
        self.case_map_by_op = {}
        self.case_map_by_file = {}

    def get_st_rpt_by_one_case(self, soc_version, op_type, case_name):
        subset_rpt = STReport()
        if op_type not in self.case_map_by_op:
            raise RuntimeError("")
        else:
            op_soc_case_map = self.case_map_by_op[op_type]
            if soc_version not in op_soc_case_map.keys():
                raise RuntimeError("")
            if case_name not in op_soc_case_map[soc_version].keys():
                raise RuntimeError("")
        subset_rpt.add_case_report(op_soc_case_map[soc_version][case_name])

    def get_st_rpt_subset(self, soc_version, op_list):
        subset_rpt = STReport()
        for op in op_list:
            if op not in self.case_map_by_op:
                raise RuntimeError("")
            else:
                op_soc_case_map = self.case_map_by_op[op]
                if soc_version not in op_soc_case_map.keys():
                    raise RuntimeError("")
                for case_rpt in op_soc_case_map[soc_version]:
                    subset_rpt.add_case_report(case_rpt)
        return subset_rpt

    def get_case_rpt(self, soc_version, op, case_name) -> STCaseReport:
        return self.case_map_by_op[op][soc_version][case_name]

    def get_generate_txt_report(self):
        result_txt_list = ["=== Text Report ==================================================================", "\n"]
        for key, val in self.case_map_by_file.items():
            result_txt_list.append("Test case file: %s" % key)
            result_txt_list.append("----------------------------------------------------------------------------------")
            for case in val:
                result_txt_list.append("  " + case.get_generate_txt_report())
            result_txt_list.append(
                "\n==================================================================================\n")
        return "\n".join(result_txt_list)

    def add_case_report(self, case_report: STCaseReport):
        if case_report.op not in self.case_map_by_op.keys():
            self.case_map_by_op[case_report.op] = {}
        print(case_report)
        print(case_report.case_file_name)
        print(self.case_map_by_file.keys())
        if case_report.case_file_name not in self.case_map_by_file.keys():
            self.case_map_by_file[case_report.case_file_name] = []
        if case_report.soc_version not in self.case_map_by_op[case_report.op].keys():
            self.case_map_by_op[case_report.op][case_report.soc_version] = {}
        self.case_map_by_op[case_report.op][case_report.soc_version][case_report.case_name] = case_report
        self.case_map_by_file[case_report.case_file_name].append(case_report)
        self.case_list.append(case_report)

    def to_json_str(self):
        json_obj = self.to_json_obj()
        return json.dumps(json_obj, indent=4)

    def to_json_obj(self):
        return [x.to_json_obj() for x in self.case_list]

    def parser_json_obj(self, json_obj):
        self.case_list = []
        self.case_map_by_op = {}
        self.case_map_by_file = {}
        for case_json_obj in json_obj:
            case_report = STCaseReport.parser_json_obj(case_json_obj)
            self.add_case_report(case_report)

    @staticmethod
    def parser_json_str(json_str):
        rpt = STReport()
        json_obj = json.loads(json_str)
        rpt.parser_json_obj(json_obj)
        return rpt


def dump_model_run_report(case_dir, report_list: List[STCaseModelRunReport]):
    reports = []
    for report in report_list:
        reports.append(report.to_json_obj())
    json_str = json.dumps(reports, indent=4)
    report_file_path = os.path.join(case_dir, "run_report.json")
    with open(report_file_path, 'w+') as r_f:
        r_f.write(json_str)


def get_model_run_report(case_dir) -> List[STCaseModelRunReport]:
    report_file_path = os.path.join(case_dir, "run_report.json")
    with open(report_file_path) as r_f:
        json_str = r_f.read()
    json_obj_list = json.loads(json_str)
    reports = []
    for json_obj in json_obj_list:
        reports.append(STCaseModelRunReport.parse_json_obj(json_obj))
    return reports


def dump_precision_report(case_dir, report_list: List[STCasePrecisionReport]):
    reports = []
    for report in report_list:
        reports.append(report.to_json_obj())
    json_str = json.dumps(reports, indent=4)
    report_file_path = os.path.join(case_dir, "precision_report.json")
    with open(report_file_path, 'w+') as r_f:
        r_f.write(json_str)


def get_precision_report(case_dir) -> List[STCasePrecisionReport]:
    report_file_path = os.path.join(case_dir, "precision_report.json")
    with open(report_file_path) as r_f:
        json_str = r_f.read()
    json_obj_list = json.loads(json_str)
    reports = []
    for json_obj in json_obj_list:
        reports.append(STCasePrecisionReport.parse_json_obj(json_obj))
    return reports


def dump_case_gen_report(case_dir, report_list: List[STCaseGenReport]):
    reports = []
    for report in report_list:
        reports.append(report.to_json_obj())
    json_str = json.dumps(reports, indent=4)
    report_file_path = os.path.join(case_dir, CASE_GEN_REPORT_FILE_NAME)
    with open(report_file_path, 'w+') as r_f:
        r_f.write(json_str)


def get_case_gen_report(case_dir) -> List[STCaseGenReport]:
    report_file_path = os.path.join(case_dir, CASE_GEN_REPORT_FILE_NAME)
    with open(report_file_path) as r_f:
        json_str = r_f.read()
    json_obj_list = json.loads(json_str)
    reports = []
    for json_obj in json_obj_list:
        reports.append(STCasePrecisionReport.parse_json_obj(json_obj))
    return reports


def get_st_gen_report(case_root_dir) -> STReport:
    st_rpt_file = os.path.join(case_root_dir, ST_GEN_REPORT_FILE_NAME)
    if not os.path.exists(st_rpt_file):
        raise RuntimeError("get st report failed, file not exist: '%s'" % st_rpt_file)
    with open(st_rpt_file) as rpt_file:
        json_str = rpt_file.read()

    return STReport.parser_json_str(json_str)


def dump_st_gen_rpt(case_root_dir, rpt: STReport):
    json_str = rpt.to_json_str()
    report_file_path = os.path.join(case_root_dir, ST_GEN_REPORT_FILE_NAME)
    with open(report_file_path, 'w+') as r_f:
        r_f.write(json_str)


def _get_time_str():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())


def dump_st_rpt(case_root_dir, rpt: STReport):
    report_file_path = os.path.join(case_root_dir, ST_RUN_REPORT_FILE_NAME)
    if os.path.exists(report_file_path):
        time_str = _get_time_str()
        report_bak_name = ST_RUN_REPORT_FILE_NAME.replace(".", time_str + ".")
        shutil.move(report_file_path, os.path.join(case_root_dir, report_bak_name))

    json_str = rpt.to_json_str()
    with open(report_file_path, 'w+') as r_f:
        r_f.write(json_str)


if __name__ == "__main__":
    s_r = STReport()
    c_r = STCaseReport()
    c_r.case_file_name = "abc"
    s_r.add_case_report(c_r)
    print(s_r.to_json_str())
    print(s_r.get_generate_txt_report())
