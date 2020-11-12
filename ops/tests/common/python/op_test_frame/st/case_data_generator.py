import os
from typing import List, Dict
from op_test_frame.st.st_case_info import STSingleCaseInfo
from op_test_frame.utils.data_generator import gen_data_file
from op_test_frame.st.st_report import STCaseDataGenReport


def _gen_case_inputs(case_dir, case_info: STSingleCaseInfo, pre_report: STCaseDataGenReport):
    input_desc_list = case_info.input_desc_list
    input_data_file_list = []
    relate_data_file_list = []
    np_data_list = []
    idx = 0
    case_data_dir = os.path.join(case_dir, "data")
    if not os.path.exists(case_data_dir):
        os.makedirs(case_data_dir)
    for input_desc in input_desc_list:
        case_input_file_name = case_info.case_name + "_input" + str(idx) + input_desc["type"] + ".bin"
        relate_data_file_list.append(os.path.join("data", case_input_file_name))
        case_input_file_path = os.path.join(case_data_dir, case_input_file_name)
        input_data_file_list.append(case_input_file_path)
        full_path = os.path.join(case_dir, case_input_file_path)
        np_array = gen_data_file(data_shape=input_desc["shape"], value_range=input_desc["value_range"],
                                 dtype=input_desc["type"], data_distribution=input_desc["data_distribute"],
                                 out_data_file=full_path)
        np_data_list.append(np_array)
        idx += 1
    pre_report.set_input_data_info(relate_data_file_list, np_data_list)


def _gen_expect_output_by_np(case_dir, case_info: STSingleCaseInfo, pre_report: STCaseDataGenReport):
    idx = 0
    np_op_args_input = []
    for input_desc in case_info.input_desc_list:
        # may be an option input
        if not input_desc or "type" not in input_desc.keys():
            np_op_args_input.append(None)
            continue
        else:
            np_op_args_input.append({
                "shape": input_desc["shape"],
                "format": input_desc["format"],
                "dtype": input_desc["type"],
                "ori_shape": input_desc["shape"] if "ori_shape" not in input_desc.keys() else input_desc["ori_shape"],
                "ori_format": input_desc["format"] if "ori_format" not in input_desc.keys() else input_desc[
                    "ori_format"],
                "value": pre_report.input_data_list[idx]
            })
            idx += 1

    expect_out_path_list = []
    relate_expect_out_file_path_list = []
    actual_out_path_list = []
    for out_idx, output_desc in enumerate(case_info.output_desc_list):
        expect_out_file_name = case_info.case_name + "_expect_np_output" + str(out_idx) + \
                               "_" + output_desc["type"] + ".bin"
        expect_out_path_list.append(os.path.join(case_dir, "data", expect_out_file_name))
        relate_expect_out_file_path_list.append(os.path.join("data", expect_out_file_name))
        out_file_name = case_info.case_name + "_output" + str(out_idx) + "_" + output_desc["type"] + ".bin"
        actual_out_path_list.append(os.path.join("data", out_file_name))
    np_op_args_output = [
        {
            "shape": desc["shape"],
            "format": desc["format"],
            "dtype": desc["type"],
            "ori_shape": desc["shape"] if "ori_shape" not in desc.keys() else desc["ori_shape"],
            "ori_format": desc["format"] if "ori_format" not in desc.keys() else desc["ori_format"],
        } for desc in case_info.output_desc_list
    ]
    np_op_args_attr = [attr["value"] for attr in case_info.attr_list]
    np_out_list = case_info.np_op(*np_op_args_input, *np_op_args_output, *np_op_args_attr)
    if not isinstance(np_out_list, (tuple, list)):
        np_out_list = [np_out_list, ]
    if len(np_out_list) > len(expect_out_path_list):
        pre_report.set_status(STCaseDataGenReport.FAILED, "np op output count not match output desc list count.")
        return
    for idx, np_out in enumerate(np_out_list):
        np_out_full_path = os.path.join(case_dir, expect_out_path_list[idx])
        np_out.tofile(np_out_full_path)
    pre_report.set_expect_out_info(expect_out_path_list, actual_out_path_list)


def generate_one_case_data(case_dir, case_info: STSingleCaseInfo):
    if case_info.np_op:
        pre_report = STCaseDataGenReport(case_info.case_name, True)
    else:
        pre_report = STCaseDataGenReport(case_info.case_name, False)

    _gen_case_inputs(case_dir, case_info, pre_report)
    if pre_report.need_gen_expect_out:
        _gen_expect_output_by_np(case_dir, case_info, pre_report)
    return pre_report


def generate_case_data_list(case_root_dir, case_info_list: List[STSingleCaseInfo]) \
        -> Dict[str, Dict[str, STCaseDataGenReport]]:
    total_report = {}
    for case_info in case_info_list:
        case_op_dir = os.path.join(case_root_dir, case_info.op)
        case_report = generate_one_case_data(case_op_dir, case_info)
        if case_info.op not in total_report.keys():
            total_report[case_info.op] = {}
        total_report[case_info.op][case_info.case_name] = case_report
    return total_report


def generate_case_data(case_info: STSingleCaseInfo, case_root_path):
    idx = 0
    np_op_args_input = []
    for input_desc in case_info.input_desc_list:
        # may be an option input
        if not input_desc or "type" not in input_desc.keys():
            np_op_args_input.append(None)
            continue
        else:
            case_input_file_path = case_info.input_data_file_paths[idx]
            full_path = os.path.join(case_root_path, case_input_file_path)
            np_array = gen_data_file(data_shape=input_desc["shape"], value_range=input_desc["value_range"],
                                     dtype=input_desc["type"],
                                     data_distribution=input_desc["data_distribute"], out_data_file=full_path)
            np_op_args_input.append({
                "shape": input_desc["shape"],
                "format": input_desc["format"],
                "dtype": input_desc["type"],
                "ori_shape": input_desc["shape"] if "ori_shape" not in input_desc.keys() else input_desc["ori_shape"],
                "ori_format": input_desc["format"] if "ori_format" not in input_desc.keys() else input_desc[
                    "ori_format"],
                "value": np_array
            })
            idx += 1

    if case_info.np_op:
        case_info.gen_np_out_data_path()
        np_op_args_output = [
            {
                "shape": desc["shape"],
                "format": desc["format"],
                "dtype": desc["type"],
                "ori_shape": desc["shape"] if "ori_shape" not in desc.keys() else desc["ori_shape"],
                "ori_format": desc["format"] if "ori_format" not in desc.keys() else desc["ori_format"],
            } for desc in case_info.output_desc_list
        ]
        np_op_args_attr = [attr["value"] for attr in case_info.attr_list]
        np_out_list = case_info.np_op(*np_op_args_input, *np_op_args_output, *np_op_args_attr)
        if not isinstance(np_out_list, (tuple, list)):
            np_out_list = [np_out_list, ]
        np_out_data_files = case_info.get_np_out_data_paths()
        for idx, np_out in enumerate(np_out_list):
            np_out_full_path = os.path.join(case_root_path, np_out_data_files[idx])
            np_out.tofile(np_out_full_path)
