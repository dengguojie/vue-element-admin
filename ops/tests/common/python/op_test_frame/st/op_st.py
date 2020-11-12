import os
import re
import zipfile
from typing import List
from op_test_frame.st import case_loader, np_op_loader, om_generator, case_data_generator, st_case_info, st_report
from op_test_frame.st import case_run_py_generator, amexec_runner, precision_comparator, case_out_runner
from op_test_frame.st.st_case_info import STSingleCaseInfo, STGenCaseInfo
from op_test_frame.st.st_report import STCaseGenReport, STReport, STCaseReport


def _find_np_op_func(op_type, np_func_map):
    np_func_map_key = re.sub(r'[^0-9a-z]', "", str("np" + op_type).lower())
    return np_func_map.get(np_func_map_key, None)


def _flesh_np_op_for_case_info(case_info_list: List[STSingleCaseInfo], np_op_map):
    for case_info in case_info_list:
        np_op = _find_np_op_func(case_info.op, np_op_map)
        case_info.np_op = np_op


def _gen_case_run_info(soc_version: str, case_list: List[STSingleCaseInfo], out_path):
    om_total_report = om_generator.generate_model_list(out_path, soc_version, case_list)
    data_total_report = case_data_generator.generate_case_data_list(out_path, case_list)
    all_op_case_run_info = {}
    total_gen_rpt_map = {}
    st_rpt = STReport()
    for case in case_list:
        st_case_rpt = STCaseReport(soc_version, case.op, case.case_name, case.case_file, STCaseReport.SUCCESS, "")
        if case.op not in total_gen_rpt_map.keys():
            total_gen_rpt_map[case.op] = []
        if case.op not in all_op_case_run_info.keys():
            all_op_case_run_info[case.op] = {}
        if soc_version not in all_op_case_run_info[case.op].keys():
            all_op_case_run_info[case.op][soc_version] = {}
        om_rpt = om_total_report[case.op][soc_version][case.case_name]
        data_rpt = data_total_report[case.op][case.case_name]
        if om_rpt.is_success() and data_rpt.is_success():
            all_op_case_run_info[case.op][soc_version][case.case_name] = \
                STGenCaseInfo(soc_version, case.op, case.case_name,
                              om_rpt.om_file_path,
                              data_rpt.input_data_file_list,
                              data_rpt.actual_out_data_file_list,
                              data_rpt.expect_out_data_file_list, None)
            case_gen_rpt = STCaseGenReport(soc_version, case.op, case.case_name, STCaseGenReport.SUCCESS)
            total_gen_rpt_map[case.op].append(case_gen_rpt)
        else:
            err_msg = ""
            if not om_rpt.is_success():
                err_msg += "Om file generate failed."
            if not data_rpt.is_success():
                err_msg += "Data gen failed. "
            case_gen_rpt = STCaseGenReport(soc_version, case.op, case.case_name, STCaseGenReport.FAILED, err_msg)
            total_gen_rpt_map[case.op].append(case_gen_rpt)

        st_case_rpt.refresh_case_gen_rpt(case_gen_rpt)
        st_rpt.add_case_report(st_case_rpt)

    for op_name, val in all_op_case_run_info.items():
        one_op_dir = os.path.join(out_path, op_name)
        st_case_info.dump_gen_case_info(one_op_dir, val)

    for op_name, val in total_gen_rpt_map.items():
        one_op_dir = os.path.join(out_path, op_name)
        st_report.dump_case_gen_report(one_op_dir, val)

    case_run_py_generator.generate_run_py_list(out_path, case_list)

    return st_rpt, total_gen_rpt_map, all_op_case_run_info


def _run_all_st(soc_version, all_op_case_run_info, out_path, st_rpt):
    case_out_runner.out_run(case_dir=out_path, soc_version=soc_version, case_gen_info_map=all_op_case_run_info,
                            st_gen_report=st_rpt)
    # for op_name, val in all_op_case_run_info.items():
    #     one_op_dir = os.path.join(out_path, op_name)
    #     amexec_rpt_list, _ = amexec_runner.run_one_op(one_op_dir, soc_version, case_info_map=val, st_rpt=st_rpt)
    #     precision_rpt_list = precision_comparator.precision_compare(one_op_dir, model_run_reports=amexec_rpt_list,
    #                                                                 case_info_map=val, st_rpt=st_rpt)


def run_st(soc_version, case_dir, np_op_dir=None, out_path="./out", run_mode="gen"):
    out_path = os.path.realpath(out_path)
    case_dir = os.path.realpath(case_dir)
    if not os.path.exists(case_dir):
        raise RuntimeError("Case file not exist, the case_file arg is: %s" % case_dir)
    if os.path.isfile(case_dir):
        case_file = case_dir
        case_list = case_loader.load_st_file(case_file)
        if not np_op_dir:
            np_op_dir = os.path.dirname(case_file)
            np_op_map = np_op_loader.load_np_op(np_op_dir)
        else:
            if os.path.isfile(np_op_dir):
                np_op_map = np_op_loader.load_np_op_file(np_op_dir)
            else:
                np_op_map = np_op_loader.load_np_op(np_op_dir)
        _flesh_np_op_for_case_info(case_list, np_op_map)
    else:
        case_list = case_loader.load_st_case(case_dir)
        if not np_op_dir:
            np_op_dir = case_dir
            np_op_map = np_op_loader.load_np_op(np_op_dir)
        else:
            if os.path.isfile(np_op_dir):
                np_op_map = np_op_loader.load_np_op_file(np_op_dir)
            else:
                np_op_map = np_op_loader.load_np_op(np_op_dir)
        _flesh_np_op_for_case_info(case_list, np_op_map)

    st_rpt, _, all_op_case_run_info = _gen_case_run_info(soc_version, case_list, out_path)
    st_report.dump_st_gen_rpt(out_path, st_rpt)
    zip_file_name = os.path.join(out_path, "st_test.zip")
    if os.path.exists(zip_file_name):
        os.remove(zip_file_name)
    zip_file = zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED)
    for path, dir_names, file_names in os.walk(out_path):
        f_path = path.replace(out_path, '')
        for filename in file_names:
            if filename == "st_test.zip":
                continue
            zip_file.write(os.path.join(path, filename), os.path.join(f_path, filename))
    zip_file.close()
    if run_mode == "run":
        _run_all_st(soc_version, all_op_case_run_info, out_path, st_rpt)
    else:
        print(st_rpt.get_generate_txt_report())
