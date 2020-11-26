import os
import json
import shutil
from typing import List, Dict
from op_test_frame.st.st_case_info import STSingleCaseInfo
from op_test_frame.st.st_report import STCaseModelGenReport


def _dump_atc_json_file(case_dir, soc_version, case_name, json_info_obj) -> str:
    json_file_dir = os.path.join(case_dir, "om", soc_version)
    if not os.path.exists(json_file_dir):
        os.makedirs(json_file_dir)
    json_file_name = case_name + ".json"
    json_str = json.dumps(json_info_obj, indent=4)
    with open(os.path.join(json_file_dir, json_file_name), 'w+') as json_f:
        json_f.write(json_str)
    return os.path.join(json_file_dir, json_file_name)


def _generate_om(case_dir, soc_version, case_name, json_file_path):
    out_path_tmp = os.path.join(case_dir, "om", soc_version, case_name + "tmp")
    out_path = os.path.join(case_dir, "om", soc_version, case_name + ".om")
    if not os.path.exists(out_path_tmp):
        os.makedirs(out_path_tmp)
    gen_om_cmd = 'atc --singleop="' + json_file_path + '" --output=' + out_path_tmp + ' --soc_version=' + soc_version
    res = os.system(gen_om_cmd)
    if res != 0:
        return False, "Gen om model failed, cmd is: %s" % gen_om_cmd, ""
    om_file = os.listdir(out_path_tmp)
    shutil.move(os.path.join(out_path_tmp, om_file[0]), out_path)
    os.removedirs(out_path_tmp)
    return True, "", out_path


def generate_model(case_op_dir, soc_version, case_info: STSingleCaseInfo) -> STCaseModelGenReport:
    json_info = [case_info.get_atc_json_obj(), ]
    json_file_path = _dump_atc_json_file(case_op_dir, soc_version, case_info.case_name, json_info)
    res, err_msg, om_file_path = _generate_om(case_op_dir, soc_version, case_info.case_name, json_file_path)
    status = STCaseModelGenReport.SUCCESS if res else STCaseModelGenReport.FAILED
    return STCaseModelGenReport(soc_version, case_info.op, case_info.case_name, status, err_msg, json_file_path, om_file_path)


def generate_model_list(case_root_dir, soc_version, case_info_list: List[STSingleCaseInfo]) \
        -> Dict[str, Dict[str, Dict[str, STCaseModelGenReport]]]:
    total_report = {}
    for case_info in case_info_list:
        case_op_dir = os.path.join(case_root_dir, case_info.op)
        case_report = generate_model(case_op_dir, soc_version, case_info)
        if case_info.op not in total_report.keys():
            total_report[case_info.op] = {}
        if soc_version not in total_report[case_info.op].keys():
            total_report[case_info.op][soc_version] = {}
        total_report[case_info.op][soc_version][case_info.case_name] = case_report
    return total_report
