import os
from typing import Dict
from op_test_frame.st import amexec_runner, precision_comparator, st_case_info, st_report
from op_test_frame.st.st_report import STReport
from op_test_frame.st.st_case_info import STGenCaseInfo
from op_test_frame.common import logger


def _check_case_dir_mod(case_dir):
    real_case_dir = os.path.realpath(case_dir)
    if not os.path.exists(real_case_dir):
        logger.log_err("The case dir is not exist, case_dir arg is: '%s', realpath is: '%s'" %
                       (case_dir, real_case_dir))
        return 0

    if not os.path.isdir(real_case_dir):
        logger.log_err("The case dir is not a directory, case_dir arg is: '%s'" % case_dir)
        return 0

    data_dir = os.path.join(real_case_dir, "data")
    if os.path.exists(data_dir):
        logger.log_err("The case dir is not case root dir, set case_dir as '%s' " % os.path.dirname(case_dir))
        return 0

    sub_dir_list = os.listdir(real_case_dir)
    for sub_dir in sub_dir_list:
        if os.path.isdir(os.path.join(real_case_dir, sub_dir)):
            sub_data_dir = os.path.join(real_case_dir, sub_dir, "data")
            if os.path.exists(sub_data_dir):
                return 2

    logger.log_err("Not find any case in this directory: case_dir arg is: '%s', realpath is: '%s'" %
                   (case_dir, real_case_dir))
    return 0


def _find_op_st_dir(case_dir, op):
    op_dir = os.path.join(case_dir, op)
    if op and os.path.exists(op_dir):
        logger.log_info("Find op st directory: '%s'" % op_dir)
        return op_dir
    file_or_dir_list = os.listdir(case_dir)
    lower_op = None if not op else str(op).lower().replace("_", "")
    for file_or_dir in file_or_dir_list:
        lower_name = file_or_dir.lower().replace("_", "")
        if lower_op and lower_name == lower_op:
            full_path = os.path.join(case_dir, file_or_dir)
            logger.log_info("Find op st directory: '%s'" % full_path)
            return full_path

    logger.log_info("Can not find op st directory, for op: '%s'" % op)
    return None


def _find_op_list_st_dir(case_dir, op_list):
    file_or_dir_list = os.listdir(case_dir)

    lower_key_path_map = {}
    for file_or_dir in file_or_dir_list:
        lower_name = file_or_dir.lower().replace("_", "")
        full_path = os.path.join(case_dir, file_or_dir)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "data")):
            lower_key_path_map[lower_name] = full_path

    find_op_list = []
    not_find_list = []
    for op in op_list:
        lower_op = None if not op else str(op).lower().replace("_", "")
        if lower_op in lower_key_path_map.keys():
            find_op_list.append((op, lower_key_path_map[lower_op]))
        else:
            not_find_list.append((op, None))
    return find_op_list, not_find_list


def _find_op_case_info(op_case_dir, op_type, st_gen_info_map: Dict[str, Dict[str, Dict[str, STGenCaseInfo]]] = None):
    if not st_gen_info_map:
        one_case_info_inner = st_case_info.get_gen_case_info(op_case_dir)
    else:
        if op_type not in st_gen_info_map.keys():
            one_case_info_inner = st_case_info.get_gen_case_info(op_case_dir)
        else:
            one_case_info_inner = st_gen_info_map[op_type]
    return one_case_info_inner


def _run_op_list(case_root_dir, soc_version, op_list, st_gen_rpt: STReport,
                 st_gen_info_map: Dict[str, Dict[str, Dict[str, STGenCaseInfo]]] = None):
    for (op_type, one_op_case_dir) in op_list:
        one_case_info = _find_op_case_info(one_op_case_dir, op_type, st_gen_info_map)
        if not one_case_info:
            logger.log_err("Not found st case gen info.")

        case_run_rpts, _ = amexec_runner.run_one_op(one_op_case_dir, soc_version,
                                                    case_info_map=one_case_info, st_rpt=st_gen_rpt)
        precision_comparator.precision_compare(one_op_case_dir, case_run_rpts,
                                               case_info_map=one_case_info, st_rpt=st_gen_rpt)

    st_report.dump_st_rpt(case_root_dir, st_gen_rpt)
    print(st_gen_rpt.get_generate_txt_report())


def _run_one_case(case_root_dir, case_dir, op_type, case_name, soc_version, st_gen_rpt: STReport,
                  st_gen_info_map: Dict[str, Dict[str, Dict[str, STGenCaseInfo]]] = None):
    one_case_info = _find_op_case_info(case_dir, op_type, st_gen_info_map)
    case_run_rpts = amexec_runner.run_one_op(case_dir, soc_version, case_name=case_name,
                                             case_info_map=one_case_info, st_rpt=st_gen_rpt)
    precision_comparator.precision_compare(case_dir, case_run_rpts,
                                           case_info_map=one_case_info, st_rpt=st_gen_rpt)
    st_report.dump_st_rpt(case_root_dir, st_report)
    print(st_gen_rpt.get_generate_txt_report())


def _build_all_case_gen_info_map(case_root_dir, soc_version):
    total_case_info_map = {}
    sub_dirs = os.listdir(case_root_dir)
    for sub_dir in sub_dirs:
        sub_dir_full = os.path.join(case_root_dir, sub_dir)
        if not os.path.isdir(sub_dir_full):
            continue
        if os.path.exists(os.path.join(sub_dir_full, st_case_info.GEN_CASE_INFO_FILE_NAME)):
            case_info_map = st_case_info.get_gen_case_info(sub_dir_full)
            if soc_version not in case_info_map.keys():
                continue
            soc_case_info_map = case_info_map[soc_version]
            for case_name, val in soc_case_info_map.items():
                op_type = val.op
            if op_type not in total_case_info_map.keys():
                total_case_info_map[op_type] = {}
                total_case_info_map[op_type][soc_version] = {}
            total_case_info_map[op_type][soc_version][case_name] = val
    return total_case_info_map


def _run_all_op_case(case_root_dir, soc_version, case_gen_info_map=None, st_gen_rpt: STReport = None):
    if not case_gen_info_map:
        case_gen_info_map = _build_all_case_gen_info_map(case_root_dir, soc_version)
    if not st_gen_rpt:
        st_gen_rpt = st_report.get_st_gen_report(case_root_dir)
    op_list = [(op_type, os.path.join(case_root_dir, op_type)) for op_type in case_gen_info_map.keys()]
    _run_op_list(case_root_dir, soc_version, op_list, st_gen_rpt, case_gen_info_map)


def out_run(case_dir, soc_version, op=None, case_name=None, op_list=None, case_gen_info_map=None,
            st_gen_report: STReport = None):
    case_dir_mode = _check_case_dir_mod(case_dir)
    if case_dir_mode == 0:
        return

    case_dir = os.path.realpath(case_dir)
    case_root_dir = case_dir
    if case_name and not op:
        logger.log_warn("'op' arg is not set, ignore 'case_name' arg.")
        case_name = None

    if op and op_list:
        logger.log_warn("'op' arg is set, ignore 'op_list' arg.")
        op_list = None

    if case_name:
        op_case_dir = _find_op_st_dir(op)
        if not st_gen_report:
            st_gen_report = st_report.get_st_gen_report(case_root_dir)
        run_set_rpt = st_gen_report.get_st_rpt_by_one_case(soc_version, op, case_name)
        if not run_set_rpt:
            return
        _run_one_case(case_root_dir, op_case_dir, op, case_name, soc_version, run_set_rpt, case_gen_info_map)
    else:
        if op:
            op_list = [op, ]
        if op_list:
            find_op_list, not_find_list = _find_op_list_st_dir(op_list)
            if not_find_list:
                logger.log_err("Some op can not found st case, not found list: [%s]" % ", ".join(not_find_list))
                return
            if not st_gen_report:
                st_gen_report = st_report.get_st_gen_report(case_root_dir)
            run_set_rpt = st_gen_report.get_st_rpt_by_one_case(soc_version, op, case_name)
            _run_op_list(case_root_dir, soc_version, find_op_list, run_set_rpt, st_gen_report)
        else:
            _run_all_op_case(case_root_dir, soc_version, case_gen_info_map, st_gen_report)
