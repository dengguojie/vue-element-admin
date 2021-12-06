#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves compile and run acl op.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import time

from op_test_frame.common import op_status

from . import utils
from .const_manager import ConstManager
from . import op_st_case_info


class AclOpRunner:
    """
    Class for compile and run acl op test code.
    """

    def __init__(self, path, soc_version, report, advance_args=None):
        self.path = path
        self.soc_version = soc_version
        self.report = report
        self.advance_args = advance_args

    def acl_compile(self):
        """
        Compile acl op
        """
        utils.print_step_log("[%s] Compile testcase test code." % (os.path.basename(__file__)))
        utils.print_info_log('Start to compile %s.' % self.path)
        cmakelist_path = os.path.join(self.path, ConstManager.CMAKE_LIST_FILE_NAME)
        if not os.path.exists(cmakelist_path):
            utils.print_error_log(
                'There is no %s in %s. Please check the path for compile.' % (
                    ConstManager.CMAKE_LIST_FILE_NAME, self.path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

        # do cmake and make
        origin_path = os.path.realpath(os.getcwd())
        build_path = os.path.join(self.path, ConstManager.BUILD_INTERMEDIATES_HOST)
        utils.check_path_valid(build_path, True)
        os.chdir(build_path)
        cmake_cmd = ['cmake', '../../..', '-DCMAKE_CXX_COMPILER=g++',
                     '-DCMAKE_SKIP_RPATH=TRUE']
        cmd_str = "cd %s && %s && %s" % (build_path, " ".join(cmake_cmd),
                                         " ".join(['make']))
        utils.print_info_log("Compile command line: %s " % cmd_str)
        main_path = os.path.join(self.path, 'run', 'out', ConstManager.MAIN)
        try:
            utils.execute_command(cmake_cmd)
            utils.execute_command(['make'])
        except utils.OpTestGenException:
            self.add_op_st_stage_result(op_status.FAILED, "compile_acl_code",
                                        None, cmd_str)
            if not os.path.exists(main_path):
                utils.print_error_log("Please check the env LD_LIBRARAY_PATH or env NPU_HOST_LIB.")
                raise utils.OpTestGenException(
                    ConstManager.ACL_COMPILE_ERROR)
        finally:
            pass
        utils.print_info_log('Finish to compile %s.' % self.path)
        os.chdir(origin_path)
        self.add_op_st_stage_result(op_status.SUCCESS, "compile_acl_code",
                                    None, cmd_str)

    def add_op_st_stage_result(self, status=op_status.FAILED,
                               stage_name=None, result=None, cmd=None):
        """
        add op st stage_result
        """
        stage_result = op_st_case_info.OpSTStageResult(
            status, stage_name, result, cmd)
        for case_report in self.report.report_list:
            case_report.trace_detail.add_stage_result(stage_result)

    def run(self):
        """
        Run acl op
        """
        main_path = os.path.join(self.path, 'run', 'out', ConstManager.MAIN)
        if not os.path.exists(main_path):
            utils.print_error_log(
                'There is no execute file "%s" in %s. Please check the path '
                'for running.' % (ConstManager.MAIN, os.path.dirname(main_path)))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        out_path = os.path.dirname(main_path)
        utils.check_path_valid(out_path, True)
        os.chdir(out_path)
        run_cmd = ['./' + ConstManager.MAIN]
        get_performance_mode = False
        if self.advance_args is not None:
            get_performance_mode = self.advance_args.get_performance_mode_flag()
        if get_performance_mode:
            utils.print_step_log("[%s] Get system performance data." % (os.path.basename(__file__)))
            prof_run_start = time.time()
            self.prof_run(out_path)
            prof_run_end = time.time()
            utils.print_info_log('System performance data executes time: %f s.'
                                 % (prof_run_end - prof_run_start))
        else:
            utils.print_step_log("[%s] Start to execute testcase." % (os.path.basename(__file__)))
            utils.print_info_log("Run command line: cd %s && %s " % (
                out_path, " ".join(run_cmd)))
            main_run_start = time.time()
            utils.execute_command(run_cmd)
            main_run_end = time.time()
            utils.print_info_log('Testcase execute in %s, cost time: %f s.'
                                 % (self.soc_version, (main_run_end - main_run_start)))
            utils.print_info_log('Finish to run %s.' % main_path)

    def prof_run(self, out_path):
        """
        use msprof to run main.
        :param out_path: path of binary main
        :return:
        """
        toolkit_root_path = os.getenv(ConstManager.INSTALL_PATH)
        if not os.path.exists(toolkit_root_path) or not toolkit_root_path:
            utils.print_error_log("Path of env install_path: "
                                  "%s does not exist" % toolkit_root_path)
            return
        if os.path.exists(toolkit_root_path):
            utils.print_info_log("Env install_path is " + toolkit_root_path)
        run_cmd = [toolkit_root_path + ConstManager.MSPROF_REL_PATH, '--application=./main',
                   '--aicpu=on', '--runtime-api=on', '--output=./' + ConstManager.PROF]
        utils.print_info_log("Run command line: cd %s && %s " % (
            out_path, " ".join(run_cmd)))
        utils.execute_command(run_cmd)
        utils.print_info_log('Finish to run main with msprof.')
        self.prof_analyze(os.path.join(out_path, ConstManager.PROF), toolkit_root_path)

    @staticmethod
    def _prof_get_op_case_info_from_csv_file(csv_file, op_name_list):
        op_case_info = []
        if not csv_file:
            utils.print_error_log("The CSV file is empty. Please check.")
            return op_case_info
        if not op_name_list:
            utils.print_error_log("The op name list is empty. Please check.")
            return op_case_info
        op_case_info = get_op_case_info_from_csv_file(
            csv_file, op_name_list)
        return op_case_info

    def _prof_get_op_name_from_report(self, run_result_list):
        op_name_list = []
        for line in run_result_list:
            if len(line.split("  ")) != ConstManager.RESULT_FILE_COLUMN_NUM:
                continue
            case_name = line.split("  ")[ConstManager.RESULT_FILE_CASE_NAME_COLUMN_NUM]
            case_report = self.report.get_case_report(case_name)
            if not case_report:
                utils.print_error_log("According case info in "
                                      "st_report.json is not found, please check")
                return []
            op_name = case_report.trace_detail.st_case_info.op_params.get(ConstManager.OP)
            if not op_name:
                utils.print_error_log("The op name got from st_report.json is empty. Please check")
                return []
            op_name_list.append(op_name)
        return op_name_list

    def _get_op_case_result_and_show_data(self, csv_file, op_name_list):
        # start to get op time from csv summary files
        op_case_info = self._prof_get_op_case_info_from_csv_file(
            csv_file, op_name_list)
        if not op_case_info:
            utils.print_error_log(
                "Failed to get the time result from CSV files. Please check.")
            return
        # show op case data.
        display_op_case_info(op_case_info)

        # start to write op time into st report
        for idx, report_obj in enumerate(self.report.report_list):
            if idx >= len(op_case_info):
                utils.print_error_log("Length of report list"
                                      " exceeds length of time result.")
                break
            prof_result = op_st_case_info.OpSTStageResult(
                op_status.SUCCESS,
                "profiling_analysis",
                op_case_info[idx][ConstManager.TASK_DURATION_INDEX] + ConstManager.PROF_TIME_UNIT)
            report_obj.trace_detail.add_stage_result(prof_result)

    def _read_result_txt(self):
        result_txt = os.path.join(self.path, ConstManager.RUN_OUT, 'result_files',
                                  'result.txt')
        if not os.path.exists(result_txt) or \
                not os.access(result_txt, os.R_OK):
            utils.print_error_log("Failed to get %s. Please check "
                                  "run result." % result_txt)
            return []

        txt = utils.read_file(result_txt)
        run_result_list = txt.split('\n')
        if len(run_result_list) <= ConstManager.NULL_RESULT_FILE_LINE_NUM:
            utils.print_error_log("Only got less than or equal to"
                                  " one line in result.txt, please check "
                                  "%s" % result_txt)
            return []
        run_result_list.pop()
        return run_result_list

    def _get_data_from_csv_summary(self, job_path, run_result_list):
        csv_file = os.path.join(job_path, ConstManager.SUMMARY_REL_PATH,
                                ConstManager.OP_SUMMARY_CSV)
        if not os.path.exists(csv_file) or \
                not os.access(csv_file, os.R_OK):
            utils.print_error_log("Failed to get %s. Please check the CSV "
                                  "summary file." % csv_file)
            return
        # start to get op names from report
        op_name_list = self._prof_get_op_name_from_report(run_result_list)
        if not op_name_list:
            utils.print_error_log(
                "Failed to get the op name from the st report. Please check.")
            return
        if op_name_list:
            utils.print_info_log(
                "Get op names from report: %s" % ','.join(op_name_list))

        # start to get op time from csv summary files and save in report
        self._get_op_case_result_and_show_data(csv_file, op_name_list)

    @staticmethod
    def _get_job_path(prof_base_path):
        scan = utils.ScanFile(prof_base_path, prefix="JOB")
        scan_dirs = scan.scan_subdirs()
        if not scan_dirs:
            utils.print_error_log("Profiling job directory"
                                  " is not found, skip according analysis")
            return ''
        if len(scan_dirs) > 1:
            utils.print_error_log(
                "Multiple profiling job directories are found, "
                "please clear the prof directory"
                " and retry: %s" % ','.join(scan_dirs))
            return ''
        job_path = os.path.join(prof_base_path, scan_dirs[0])
        os.chdir(job_path)
        utils.print_info_log(
            "Start to analyze profiling data in %s" % job_path)
        return job_path

    def prof_analyze(self, prof_base_path, toolkit_root_path):
        """
        do profiling analysis.
        :param prof_base_path: base path of profiling data: run/out/prof
        :param toolkit_root_path: installed path of toolkit package
        :return:
        """
        try:
            job_path = self._get_job_path(prof_base_path)
            if not job_path:
                return
            # start to read result.txt and get op execute times
            run_result_list = self._read_result_txt()
            if not run_result_list:
                return
            # start to do export summary
            analyze_cmd = [ConstManager.PROF_PYTHON_CMD,
                           toolkit_root_path + ConstManager.MSPROF_PYC_REL_PATH,
                           'export', 'summary',
                           '-dir=./']
            utils.execute_command(analyze_cmd)
            self._get_data_from_csv_summary(job_path, run_result_list)
        except IOError:
            utils.print_error_log("Operate directory of profiling data failed")
        finally:
            pass


def _get_op_case_info_list(column_line_list, row_list, op_name_list, op_case_info_list):
    each_case_info_list = []
    task_id_column_idx = column_line_list.index(
        ConstManager.OP_CASE_INFO_IN_CSV_COLUMN_NAME_LIST[
            ConstManager.TASK_ID_INDEX])
    op_time_column_idx = column_line_list.index(
        ConstManager.OP_CASE_INFO_IN_CSV_COLUMN_NAME_LIST[
            ConstManager.TASK_DURATION_INDEX])
    op_name_column_idx = column_line_list.index(
        ConstManager.OP_CASE_INFO_IN_CSV_COLUMN_NAME_LIST[
            ConstManager.OP_NAME_INDEX])
    task_type_column_idx = column_line_list.index(
        ConstManager.OP_CASE_INFO_IN_CSV_COLUMN_NAME_LIST[
            ConstManager.TASK_TYPE_INDEX])
    row_list_sorted = sorted(row_list,
                             key=lambda x: int(x[task_id_column_idx]))

    op_idx = 0
    for _, row in enumerate(row_list_sorted):
        if op_idx == len(op_name_list):
            break
        if op_name_list[op_idx] in os.path.split(
                row[op_name_column_idx])[1]:
            op_time = row[op_time_column_idx]
            op_type = row[op_name_column_idx]
            op_idx = op_idx + 1
            each_case_info_list.extend([op_type,  row[task_type_column_idx],
                                        op_time.strip('"')])
            op_case_info_list.append(each_case_info_list)
            each_case_info_list = []
    return op_case_info_list


def get_op_case_info_from_csv_file(csv_file, op_name_list):
    """
    get op case info from csv file
    """
    op_case_info_list = []
    with open(csv_file, 'r') as csv_file_obj:
        try:
            import csv
        except ImportError as import_error:
            utils.print_error_log(
                "[acl_op_runner] Unable to import the CSV file: %s. Please check."
                % str(import_error))
            return op_case_info_list
        finally:
            pass
        row_list = list(csv.reader(csv_file_obj))
        if not row_list:
            utils.print_error_log("The CSV summary file is empty. Please check.")
            return op_case_info_list
        column_line_list = row_list.pop(0)  # remove column line
        for each_case_iter in ConstManager.OP_CASE_INFO_IN_CSV_COLUMN_NAME_LIST:
            if each_case_iter not in column_line_list:
                utils.print_error_log("%s not found in the column line. Please check."
                                      % each_case_iter)
                return op_case_info_list
        op_case_info_list = _get_op_case_info_list(
            column_line_list, row_list, op_name_list, op_case_info_list)
    return op_case_info_list


def display_op_case_info(op_case_info_list):
    """
    display_op_case_info
    """
    utils.print_info_log(
        '---------------------------------------------------')
    utils.print_info_log(
        'OP Type \t Task Type \t Task Duration(us)')
    utils.print_info_log(
        '---------------------------------------------------')
    op_case_count = len(op_case_info_list)
    if op_case_count <= ConstManager.SHOW_DATA_UPPER_LIMLT:
        for i in range(op_case_count):
            utils.print_info_log('%s \t %s \t %f' % (
                op_case_info_list[i][ConstManager.OP_NAME_INDEX], op_case_info_list[i][ConstManager.TASK_TYPE_INDEX],
                float(op_case_info_list[i][ConstManager.TASK_DURATION_INDEX])))
    else:
        for i in range(ConstManager.SHOW_TOP_TEN_DATA):
            utils.print_info_log('%s \t %s \t %f' % (
                op_case_info_list[i][ConstManager.OP_NAME_INDEX], op_case_info_list[i][ConstManager.TASK_TYPE_INDEX],
                float(op_case_info_list[i][ConstManager.TASK_DURATION_INDEX])))
        utils.print_info_log('...   \t   ...   \t   ...')
        for i in range(op_case_count - ConstManager.SHOW_LAST_TEN_DATA, op_case_count):
            utils.print_info_log('%s \t %s \t %f' % (
                op_case_info_list[i][ConstManager.OP_NAME_INDEX], op_case_info_list[i][ConstManager.TASK_TYPE_INDEX],
                float(op_case_info_list[i][ConstManager.TASK_DURATION_INDEX])))
