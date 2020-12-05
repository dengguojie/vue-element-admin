#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves compile and run acl op.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved Â© 2020
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
TEST_PY = 'test_{op_name}.py'


class MsOpRunner:
    """
    Class for compile and run acl op test code.
    """

    def __init__(self, path, op_name, soc_version, report):
        self.path = path
        self.soc_version = soc_version
        self.report = report
        self.op_name = op_name

    @staticmethod
    def _execute_command(cmd):
        utils.print_info_log('Execute command: %s' % cmd)
        process = subprocess.Popen(cmd.split(), shell=False,
                                   stdout=subprocess.PIPE,
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

    def run(self):
        """
        Run acl op
        """
        # test_data
        test_file = TEST_PY.format(op_name=self.op_name)
        test_py_path = os.path.join(self.path, 'src', test_file)
        utils.print_info_log('Start to run %s.' % test_py_path)
        if not os.path.exists(test_py_path):
            utils.print_error_log(
                'There is no execute file "%s" in %s. Please check the path '
                'for running.' % (test_file, os.path.dirname(test_py_path)))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        src_path = os.path.dirname(test_py_path)
        utils.check_path_valid(src_path, True)
        os.chdir(src_path)
        run_cmd = 'pytest -s ' + test_file
        utils.print_info_log("Run command line: cd %s && %s " % (
            src_path, run_cmd))
        self._execute_command(run_cmd)
        utils.print_info_log('Finish to run %s.' % test_py_path)

    def process(self):
        """
        run mindspore op
        """
        self.run()
