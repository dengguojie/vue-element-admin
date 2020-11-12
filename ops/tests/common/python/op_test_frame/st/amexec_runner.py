import os
import shutil
from typing import List, Dict
from op_test_frame.st import st_report
from op_test_frame.st.st_report import STCaseModelRunReport, STReport
from op_test_frame.st.st_case_info import STGenCaseInfo
from op_test_frame.st import st_case_info
from op_test_frame.common import logger


def _fix_amexec_out_path(out_tmp_dir, case_dir, case_info: STGenCaseInfo):
    if not os.path.exists(out_tmp_dir):
        return False
    tmp_subs = os.listdir(out_tmp_dir)
    tmp_subs = sorted(tmp_subs)
    tmp_sub = tmp_subs[-1]
    data_list = os.listdir(os.path.join(out_tmp_dir, tmp_sub))
    out_data_file_list = case_info.output_files
    if len(data_list) != len(out_data_file_list):
        logger.log_err("Case Name: %s, amexec out data count not equal expect out data count" % case_info.case_name)
        return False, "amexec out data count not equal expect out data count"
    for data_file, out_file in zip(data_list, out_data_file_list):
        data_file_full_path = os.path.join(out_tmp_dir, tmp_sub, data_file)
        shutil.move(data_file_full_path, os.path.join(case_dir, out_file))
    return True, ""


def _run_one_case(case_dir: str, case_info: STGenCaseInfo) -> STCaseModelRunReport:
    input_files = [os.path.join(case_dir, input_file) for input_file in case_info.input_files]
    om_file_path = os.path.join(case_dir, case_info.om_file)
    output_tmp_dir = os.path.join(case_dir, "amexec_out")
    if not os.path.exists(output_tmp_dir):
        os.makedirs(output_tmp_dir)
    amexec_cmd = "amexec --model %s --input %s --output %s --outfmt BIN --loop 1" % (
        om_file_path, ",".join(input_files), output_tmp_dir)
    res = os.system(amexec_cmd)
    if res != 0:
        return STCaseModelRunReport(case_info.soc_version, case_info.op, case_info.case_name,
                                    STCaseModelRunReport.FAILED, "Amexec Failed")
    else:
        res = _fix_amexec_out_path(output_tmp_dir, case_dir, case_info)
        if not res:
            return STCaseModelRunReport(case_info.soc_version, case_info.op, case_info.case_name,
                                        STCaseModelRunReport.FAILED, "not found amexec out data file")
        else:
            return STCaseModelRunReport(case_info.soc_version, case_info.op, case_info.case_name,
                                        STCaseModelRunReport.SUCCESS)


def run_one_op(case_dir, soc_version, case_name=None, case_info_map: Dict[str, Dict[str, STGenCaseInfo]] = None,
               st_rpt: STReport = None) -> List[STCaseModelRunReport]:
    if not case_info_map:
        soc_case_map = st_case_info.get_gen_case_info(case_dir)
    else:
        soc_case_map = case_info_map
    if soc_version not in soc_case_map.keys():
        logger.log_err("not support this soc version: %s. case dir: %s" % (soc_version, case_dir))
        return [], soc_case_map
    soc_case_list = soc_case_map[soc_version]
    if not soc_case_list:
        logger.log_err("Has no case in this soc version: %s. case dir: %s" % (soc_version, case_dir))
        return [], soc_case_map
    report_list = []
    if not case_name:
        logger.log_info("start run all case in :%s" % case_dir)
        for key, val in soc_case_list.items():
            logger.log_info("start run case: %s" % key)
            case_report = _run_one_case(case_dir, val)
            if st_rpt:
                rpt = st_rpt.get_case_rpt(soc_version, val.op, val.case_name)
                if rpt:
                    rpt.refresh_case_run_rpt(case_report)
            report_list.append(case_report)
    else:
        if case_name not in soc_case_list.keys():
            logger.log_err("case name:'%s' is not in this case dir: '%s'" % (case_name, case_dir))
            return [], soc_case_map
        case_report = _run_one_case(case_dir, soc_case_list[case_name])
        if st_rpt:
            rpt = st_rpt.get_case_rpt(soc_version, soc_case_list[case_name].op, soc_case_list[case_name].case_name)
            if rpt:
                rpt.refresh_case_run_rpt(case_report)
        report_list.append(case_report)
    st_report.dump_model_run_report(case_dir, report_list)
    return report_list, soc_case_map
