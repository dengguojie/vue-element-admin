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
        utils.print_info_log("Run command line: cd %s && %s " % (
            out_path, " ".join(run_cmd)))
        self._execute_command(run_cmd)
        utils.print_info_log('Finish to run %s.' % main_path)

    def process(self):
        """
        compile and run acl op
        """
        self.acl_compile()
        self.run()
