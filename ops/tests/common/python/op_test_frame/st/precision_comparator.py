import os
from typing import Dict, List
from op_test_frame.st import st_report
from op_test_frame.st.st_report import STCasePrecisionReport, STCaseModelRunReport, STReport
from op_test_frame.st.st_case_info import STGenCaseInfo
from op_test_frame.utils import precision_compare_util
from op_test_frame.common import logger


def _check_precision(case_dir, case_info: STGenCaseInfo) -> STCasePrecisionReport:
    output_files = case_info.output_files
    expect_output_files = case_info.expect_outfiles
    status = STCasePrecisionReport.SUCCESS
    err_msg = "Not need compare precision"
    if not case_info.expect_outfiles:
        logger.log_info("Case name: %s, status: %s, msg: %s" % (case_info.case_name, "SUCCESS", err_msg))
        return STCasePrecisionReport(case_info.soc_version, case_info.op, case_info.case_name,
                                     STCasePrecisionReport.SUCCESS, err_msg)
    for output_file, expect_output_file in zip(output_files, expect_output_files):
        output_file_full_path = os.path.join(case_dir, output_file)
        expect_output_file_full_path = os.path.join(case_dir, expect_output_file)
        result = precision_compare_util.compare_precision(output_file_full_path, expect_output_file_full_path,
                                                          case_info.precision_standard)
        if result.status != "SUCCESS":
            status = STCasePrecisionReport.FAILED
            err_msg = "Compare actual_output: '%s' with expect_output: '%s' failed, error info: %s" % (
                output_file, expect_output_file, result.err_msg)
        logger.log_info(
            "Case name: %s, output check status: %s, msg: %s" % (case_info.case_name, result.status, result.err_msg))
    return STCasePrecisionReport(case_info.soc_version, case_info.op, case_info.case_name, status, err_msg)


def precision_compare(case_dir, model_run_reports: List[STCaseModelRunReport] = None,
                      case_info_map: Dict[str, Dict[str, STGenCaseInfo]] = None, st_rpt: STReport = None):
    logger.log_info("Run precision comparator start, case dir: %s, case count: %d" % (case_dir, len(model_run_reports)))
    precision_report_list = []
    success_cnt = 0
    err_cnt = 0
    for model_run_report in model_run_reports:
        if model_run_report.is_success():
            case_info = case_info_map[model_run_report.soc_version][model_run_report.case_name]
            report = _check_precision(case_dir, case_info)

            if st_rpt:
                one_rpt = st_rpt.get_case_rpt(report.soc_version, report.op, report.case_name)
                if one_rpt:
                    one_rpt.refresh_case_precision_rpt(report)

            precision_report_list.append(report)
            if report.is_success():
                success_cnt += 1
            else:
                err_cnt += 1
        else:
            logger.log_info("Case name: %s, model run failed, can not compare precision" % model_run_report.case_name)
    st_report.dump_precision_report(case_dir, precision_report_list)

    logger.log_info("Run precision comparator end, case dir: %s, success count: %d, error count: %d" % (
        case_dir, success_cnt, err_cnt))
    return precision_report_list
