import os
import json
import re
from typing import List
from op_test_frame.st.st_case_info import STMultiCaseInfo, STSingleCaseInfo


def _parse_json_file(json_file):
    with open(json_file) as f:
        json_str = f.read()
    json_objs = json.loads(json_str)
    case_idx = 0
    total_case_in_file = []
    for json_obj in json_objs:
        json_obj_keys = json_obj.keys()
        if "op" not in json_obj_keys:
            raise RuntimeError(
                "parse test case json file failed, case has no \"op\" info, case json file: %s" % json_file)

        input_desc_list = [] if "input_desc" not in json_obj_keys else json_obj["input_desc"]
        output_desc_list = [] if "output_desc" not in json_obj_keys else json_obj["output_desc"]
        attr_list = [] if "attr" not in json_obj_keys else json_obj["attr"]
        op = json_obj["op"]
        if "case_name" in json_obj_keys:
            case_name = json_obj["case_name"]
        else:
            case_name = op + "_" + os.path.basename(json_file) + "_" + str(case_idx)

        is_multi = False

        for input_desc in input_desc_list:
            if isinstance(input_desc["format"], (tuple, list)):
                is_multi = True
            if isinstance(input_desc["shape"][0], (tuple, list)):
                is_multi = True
            if isinstance(input_desc["type"][0], (tuple, list)):
                is_multi = True
            if "value_range" in input_desc.keys() and isinstance(input_desc["value_range"][0], (tuple, list)):
                is_multi = True
            if "data_distribute" in input_desc.keys() and isinstance(input_desc["data_distribute"], (tuple, list)):
                is_multi = True
            if "data_file" in input_desc.keys() and isinstance(input_desc["data_file"], (tuple, list)):
                is_multi = True
        for output_desc in output_desc_list:
            if isinstance(output_desc["format"], (tuple, list)):
                is_multi = True
            if isinstance(output_desc["shape"][0], (tuple, list)):
                is_multi = True
            if isinstance(output_desc["type"][0], (tuple, list)):
                is_multi = True
        for attr_info in attr_list:
            if isinstance(attr_info["value"], (tuple, list)):
                is_multi = True
        if is_multi:
            multi_case = STMultiCaseInfo(case_name, op, input_desc_list, output_desc_list, attr_list, json_file,
                                         case_idx)
            for single_case in multi_case.gen_single_cases():
                total_case_in_file.append(single_case)
        else:
            single_case = STSingleCaseInfo(case_name, op, input_desc_list, output_desc_list, attr_list, json_file,
                                           case_idx)
            total_case_in_file.append(single_case)
        case_idx += 1
    return total_case_in_file


def _scan_case_json(case_dir):
    case_files = os.listdir(case_dir)
    case_json_list = []
    for case_file in case_files:
        if os.path.isdir(os.path.join(case_dir, case_file)):
            if case_file.endswith(os.path.sep + "out"):
                # is st out tmp file, ignore it
                continue
            else:
                case_json_list_in_sub = _scan_case_json(os.path.join(case_dir, case_file))
                for case_json in case_json_list_in_sub:
                    case_json_list.append(case_json)
        else:
            if re.match(r'.*_case.*\.json$', case_file):
                case_json_list.append(os.path.join(case_dir, case_file))
    return case_json_list


def load_st_file(case_file) -> List[STSingleCaseInfo]:
    total_case = []
    json_file_list = [case_file]
    for json_file in json_file_list:
        case_list = _parse_json_file(json_file)
        for case in case_list:
            total_case.append(case)
    return total_case


def load_st_case(case_dir) -> List[STSingleCaseInfo]:
    json_file_list = _scan_case_json(case_dir)
    total_case = []
    for json_file in json_file_list:
        case_list = _parse_json_file(json_file)
        for case in case_list:
            total_case.append(case)
    return total_case
