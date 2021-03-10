#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves compile and run acl op.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import sys
    import os
    import subprocess
    from . import utils
    from op_test_frame.common import op_status
    from . import op_st_case_info
except ImportError as import_error:
    sys.exit("[acl_op_runner] Unable to import module: %s." % str(
        import_error))

CMAKE_LIST_FILE_NAME = 'CMakeLists.txt'
BUILD_INTERMEDIATES_HOST = 'build/intermediates/host'
RUN_OUT = 'run/out'
MAIN = 'main'
PROF = 'prof'
INSTALL_PATH = 'install_path'
MSPROF_REL_PATH = '/toolkit/tools/profiler/bin/msprof'
MSPROF_PYC_REL_PATH = '/toolkit/tools/profiler/profiler_tool/analysis/msprof/msprof.pyc'
SUMMARY_REL_PATH = 'summary'
TASK_TIME_CSV = 'task_time_0_1.csv'
PROF_PYTHON_CMD = "python3.7"
CSV_OP_NAME_COLUMN_NUM = 12
CSV_TASK_ID_COLUMN_NUM = 11
CSV_TIME_COLUMN_NUM = 1
NULL_RESULT_FILE_LINE_NUM = 2
RESULT_FILE_COLUMN_NUM = 3
RESULT_FILE_CASE_NAME_COLUMN_NUM = 1
PROF_TIME_UNIT = 'us'
CSV_TASK_ID_COLUMN_NAME = 'Task ID'
CSV_OP_TIME_COLUMN_NAME = 'Time(us)'
CSV_OP_NAME_COLUMN_NAME = 'Op Name'
OP = 'op'


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
        utils.print_info_log('Start to compile %s.' % self.path)
        cmakelist_path = os.path.join(self.path, CMAKE_LIST_FILE_NAME)
        if not os.path.exists(cmakelist_path):
            utils.print_error_log(
                'There is no %s in %s. Please check the path for compile.' % (
                    CMAKE_LIST_FILE_NAME, self.path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)

        # do cmake and make
        build_path = os.path.join(self.path, BUILD_INTERMEDIATES_HOST)
        utils.check_path_valid(build_path, True)
        os.chdir(build_path)
        cmake_cmd = ['cmake', '../../..', '-DCMAKE_CXX_COMPILER=g++',
                     '-DCMAKE_SKIP_RPATH=TRUE']
        make_cmd = ['make']
        cmd_str = "cd %s && %s && %s" % (build_path, " ".join(cmake_cmd),
                                         " ".join(make_cmd))
        utils.print_info_log("Compile command line: %s " % cmd_str)
        try:
            self._execute_command(cmake_cmd)
            self._execute_command(make_cmd)
        except utils.OpTestGenException:
            self.add_op_st_stage_result(op_status.FAILED, "compile_acl_code",
                                        None, cmd_str)
        utils.print_info_log('Finish to compile %s.' % self.path)
        self.add_op_st_stage_result(op_status.SUCCESS, "compile_acl_code",
                                    None, cmd_str)
        # set atc & acl log level env.
        self.set_log_level_env()
        # do atc single op model conversion
        utils.print_info_log('Start to convert single op.')
        run_out_path = os.path.join(self.path, RUN_OUT)
        os.chdir(run_out_path)
        atc_cmd = self._get_atc_cmd()
        cmd_str = "cd %s && %s " % (run_out_path, " ".join(atc_cmd))
        utils.print_info_log("ATC command line: %s" % cmd_str)
        try:
            self._execute_command(atc_cmd)
        except utils.OpTestGenException:
            self.add_op_st_stage_result(op_status.FAILED,
                                        "atc_single_op_convert",
                                        None, cmd_str)
        self.add_op_st_stage_result(op_status.SUCCESS,
                                    "atc_single_op_convert",
                                    None, cmd_str)
        utils.print_info_log('Finish to convert single op.')

    def add_op_st_stage_result(self, status=op_status.FAILED,
                               stage_name=None, result=None, cmd=None):
        stage_result = op_st_case_info.OpSTStageResult(
            status, stage_name, result, cmd)
        for case_report in self.report.report_list:
            case_report.trace_detail.add_stage_result(stage_result)

    def _get_atc_cmd(self):
        atc_cmd = ['atc', '--singleop=test_data/config/acl_op.json',
                   '--soc_version=' + self.soc_version, '--output=op_models']
        if self.advance_args is not None:
            atc_advance_cmd = self.advance_args.get_atc_advance_cmd()
            atc_cmd.extend(atc_advance_cmd)
        return atc_cmd

    @staticmethod
    def _execute_command(cmd):
        utils.print_info_log('Execute command: %s' % cmd)
        process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        while process.poll() is None:
            line = process.stdout.readline()
            line = line.strip()
            if line:
                print(line)
        if process.returncode != 0:
            utils.print_error_log('Failed to execute command: %s' % cmd)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def set_log_level_env(self):
        if self.advance_args is not None:
            utils.print_info_log('Set env for ATC & ACL.')
            get_log_level, get_slog_flag = self.advance_args.get_env_value()
            set_log_level_env = ['export', 'ASCEND_GLOBAL_LOG_LEVEL='
                                 + get_log_level]
            set_slog_print_env = ['export', 'ASCEND_SLOG_PRINT_TO_STDOUT='
                                  + get_slog_flag]
            utils.print_info_log("Set env command line: %s && %s " % (
                " ".join(set_log_level_env), " ".join(set_slog_print_env)))
            os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = get_log_level
            os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = get_slog_flag
            utils.print_info_log('Finish to set env for ATC & ACL.')
        return

    def run(self):
        """
        Run acl op
        """
        main_path = os.path.join(self.path, 'run', 'out', MAIN)
        utils.print_info_log('Start to run %s.' % main_path)
        if not os.path.exists(main_path):
            utils.print_error_log(
                'There is no execute file "%s" in %s. Please check the path '
                'for running.' % (MAIN, os.path.dirname(main_path)))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        out_path = os.path.dirname(main_path)
        utils.check_path_valid(out_path, True)
        os.chdir(out_path)
        run_cmd = ['./' + MAIN]
        get_performance_mode = False
        if self.advance_args is not None:
            get_performance_mode = self.advance_args.get_performance_mode_flag()
        if get_performance_mode:
            self.prof_run(out_path)
        else:
            utils.print_info_log("Run command line: cd %s && %s " % (
                out_path, " ".join(run_cmd)))
            self._execute_command(run_cmd)
            utils.print_info_log('Finish to run %s.' % main_path)

    def prof_run(self, out_path):
        """
        use msprof to run main.
        :param out_path: path of binary main
        :return:
        """
        toolkit_root_path = os.getenv(INSTALL_PATH)
        if not os.path.exists(toolkit_root_path):
            utils.print_error_log("Path of env install_path: "
                                  "%s does not exist" % toolkit_root_path)
            return
        else:
            utils.print_info_log("Env install_path is " + toolkit_root_path)
        run_cmd = [toolkit_root_path + MSPROF_REL_PATH,
                   '--application=./main', '--output=./' + PROF]
        utils.print_info_log("Run command line: cd %s && %s " % (
            out_path, " ".join(run_cmd)))
        self._execute_command(run_cmd)
        utils.print_info_log('Finish to run main with msprof.')
        self.prof_analyze(os.path.join(out_path, PROF), toolkit_root_path)

    @staticmethod
    def _prof_get_op_time_from_csv_file(csv_file, op_name_list):
        time_result = []
        if not csv_file:
            utils.print_error_log("Csv file is empty, please check.")
            return time_result
        if not op_name_list:
            utils.print_error_log("Op name list is empty, please check.")
            return time_result
        with open(csv_file, 'r') as csv_file_obj:
            try:
                import csv
            except import_error:
                utils.print_error_log("[acl_op_runner] Unable to import csv, please check.")
                return time_result
            csv_reader = csv.reader(csv_file_obj)
            row_list = list(csv_reader)
            if not row_list:
                utils.print_error_log("Csv summary file is empty, please check.")
                return time_result
            column_line_list = row_list.pop(0)  # remove column line
            if CSV_TASK_ID_COLUMN_NAME not in column_line_list or \
                    CSV_OP_TIME_COLUMN_NAME not in column_line_list or \
                    CSV_OP_NAME_COLUMN_NAME not in column_line_list:
                utils.print_error_log("Task ID , Op Name or Time(us) not found in column line, please check.")
                return time_result
            task_id_column_idx = column_line_list.index(CSV_TASK_ID_COLUMN_NAME)
            op_time_column_idx = column_line_list.index(CSV_OP_TIME_COLUMN_NAME)
            op_name_column_idx = column_line_list.index(CSV_OP_NAME_COLUMN_NAME)
            row_list = sorted(row_list, key=lambda x: int(x[task_id_column_idx]))

            op_idx = 0
            for row_idx in range(0, len(row_list)):
                if op_idx == len(op_name_list):
                    break
                if op_name_list[op_idx] in os.path.split(
                        row_list[row_idx][op_name_column_idx])[1]:
                    op_time = row_list[row_idx][op_time_column_idx]
                    op_idx = op_idx + 1
                    time_result.append(op_time)
        return time_result

    def _prof_get_op_name_from_report(self, run_result_list):
        op_name_list = []
        for line in run_result_list:
            if len(line.split("  ")) != RESULT_FILE_COLUMN_NUM:
                continue
            case_name = line.split("  ")[RESULT_FILE_CASE_NAME_COLUMN_NUM]
            case_report = self.report.get_case_report(case_name)
            if not case_report:
                utils.print_error_log("According case info in "
                                      "st_report.json is not found, please check")
                return []
            op_name = case_report.trace_detail.st_case_info.op_params.get(OP)
            if not op_name:
                utils.print_error_log("Op name got from st_report.json is empty, please check")
                return []
            op_name_list.append(op_name)
        return op_name_list

    def prof_analyze(self, prof_base_path, toolkit_root_path):
        """
        do profiling analysis.
        :param prof_base_path: base path of profiling data: run/out/prof
        :param toolkit_root_path: installed path of toolkit package
        :return:
        """
        try:
            scan = utils.ScanFile(prof_base_path, prefix="JOB")
            scan_dirs = scan.scan_subdirs()
            if not scan_dirs:
                utils.print_error_log("Profiling job directory"
                                      " is not found, skip according analysis")
                return
            if len(scan_dirs) > 1:
                utils.print_error_log("Multiple profiling job directories are found, "
                                      "please clear the prof directory"
                                      " and retry: %s" % ','.join(scan_dirs))
                return
            job_path = os.path.join(prof_base_path, scan_dirs[0])
            os.chdir(job_path)
            utils.print_info_log("Start to analyze profiling data in %s" % job_path)

            # start to read result.txt and get op execute times
            result_txt = os.path.join(self.path, RUN_OUT, 'result_files',
                                      'result.txt')
            if not os.path.exists(result_txt) or \
                    not os.access(result_txt, os.R_OK):
                utils.print_error_log("Failed to get %s, please check "
                                      "run result." % result_txt)
                return

            txt = utils.read_file(result_txt)
            run_result_list = txt.split('\n')
            if len(run_result_list) <= NULL_RESULT_FILE_LINE_NUM:
                utils.print_error_log("Only got less than or equal to"
                                      " one line in result.txt, please check "
                                      "%s" % result_txt)
                return
            run_result_list.pop()

            # start to do export summary
            analyze_cmd = [PROF_PYTHON_CMD,
                           toolkit_root_path + MSPROF_PYC_REL_PATH,
                           'export', 'summary',
                           '-dir=./']
            self._execute_command(analyze_cmd)

            csv_file = os.path.join(job_path, SUMMARY_REL_PATH, TASK_TIME_CSV)
            if not os.path.exists(csv_file) or \
                    not os.access(csv_file, os.R_OK):
                utils.print_error_log("Failed to get %s, please check "
                                      "summary csv file." % csv_file)
                return
            # start to get op names from report
            op_name_list = self._prof_get_op_name_from_report(run_result_list)
            if not op_name_list:
                utils.print_error_log("Failed to get op names from st report, please check.")
                return
            else:
                utils.print_info_log("Get op names from report: %s" % ','.join(op_name_list))

            # start to get op time from csv summary files
            time_result = self._prof_get_op_time_from_csv_file(csv_file, op_name_list)
            if not time_result:
                utils.print_error_log("Failed to get time result from csv files, please check.")
                return
            utils.print_info_log("Get time cost of each case from csv: %s" % ','.join(time_result))

            # start to write op time into st report
            for idx in range(0, len(self.report.report_list)):
                if idx >= len(time_result):
                    utils.print_error_log("Length of report list"
                                          " exceeds length of time result.")
                    break
                prof_result = op_st_case_info.OpSTStageResult(
                    op_status.SUCCESS,
                    "profiling_analysis",
                    time_result[idx] + PROF_TIME_UNIT)
                self.report.report_list[idx].trace_detail.add_stage_result(prof_result)
            utils.print_info_log("Finished to analyze profiling data")
        except IOError:
            utils.print_error_log("Operate directory of profiling data failed")

    def process(self):
        """
        compile and run acl op
        """
        self.acl_compile()
        self.run()
