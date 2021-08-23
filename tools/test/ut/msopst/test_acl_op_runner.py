import unittest
import pytest
import os
import sys
from unittest import mock
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.st_report import OpSTReport
from op_test_frame.st.interface.acl_op_runner import AclOpRunner
from op_test_frame.st.interface.advance_ini_parser import AdvanceIniParser

sys.path.append(os.path.dirname(__file__) + "/../../")
MSOPST_CONF_INI = './msopst/golden/base_case/input/msopst.ini'

class Process():
    def __init__(self, return_code=1):
        self.returncode = return_code

    def poll(self):
        return "None"

class TestUtilsMethods(unittest.TestCase):
    def test_msopst_compile1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.exists', return_value=False):
                runner = AclOpRunner('/home', 'ddd', 'report')
                runner.acl_compile()
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_msopst_compile2(self):
        report = OpSTReport()
        process = Process()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch('os.path.exists', return_value=True), mock.patch('os.chdir'):
                with mock.patch('subprocess.Popen', return_value=process):
                    runner = AclOpRunner('/home', 'ddd', report)
                    runner.acl_compile()

    def test_msopst_compile3(self):
        report = OpSTReport()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner.AclOpRunner._execute_command'):
                with mock.patch('os.path.exists', return_value=True), mock.patch('os.chdir'):
                    runner = AclOpRunner('/home', 'ddd', report)
                    runner.acl_compile()

    def test_msopst_run1(self):
        report = OpSTReport()
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.exists', return_value=False):
                runner = AclOpRunner('/home', 'ddd', report)
                runner.run()
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_msopst_run2(self):
        report = OpSTReport()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner.AclOpRunner._execute_command'):
                with mock.patch('os.path.exists',
                                return_value=True), mock.patch('os.chdir'):
                    runner = AclOpRunner('/home', 'ddd', report)
                    runner.run()

    def test_msopst_run3(self):
        report = OpSTReport()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner.AclOpRunner._execute_command'):
                with mock.patch('os.path.exists',
                                return_value=True), mock.patch('os.chdir'):
                    runner = AclOpRunner('/home', 'ddd', report)
                    runner.process()

    def test_msopst_run4(self):
        report = OpSTReport()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner.AclOpRunner._execute_command'):
                with mock.patch('os.path.exists',
                                return_value=True), mock.patch('os.chdir'):
                    advance = AdvanceIniParser(MSOPST_CONF_INI)
                    runner = AclOpRunner('/home', 'ddd', report, advance)
                    runner.process()

    def test_msopst_get_atc_cmd(self):
        report = OpSTReport()
        advance = AdvanceIniParser(MSOPST_CONF_INI)
        runner = AclOpRunner('/home', 'ddd', report, advance)
        runner._get_atc_cmd()

    def test_msopst_set_log_level_env(self):
        report = OpSTReport()
        advance = AdvanceIniParser(MSOPST_CONF_INI)
        runner = AclOpRunner('/home', 'ddd', report, advance)
        runner.set_log_level_env()

    def test_msopst_prof_run1(self):
        with mock.patch('os.getenv', return_value="/home"):
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch(
                        'op_test_frame.st.interface.acl_op_runner.AclOpRunner._execute_command'):
                    report = OpSTReport()
                    advance = AdvanceIniParser(MSOPST_CONF_INI)
                    runner = AclOpRunner('/home', 'ddd', report, advance)
                    runner.prof_run('/home')

    def test_msopst_prof_run2(self):
        with mock.patch('os.getenv', return_value="/home"):
            with mock.patch('os.path.exists', return_value=False):
                report = OpSTReport()
                advance = AdvanceIniParser(MSOPST_CONF_INI)
                runner = AclOpRunner('/home', 'ddd', report, advance)
                runner.prof_run('/home')

    def test_msopst_prof_get_op_time_from_csv_file_1(self):
        report = OpSTReport()
        runner = AclOpRunner('/home', 'ddd', report)
        runner._prof_get_op_case_info_from_csv_file(None, ["add"])
        runner._prof_get_op_case_info_from_csv_file("file", None)


if __name__ == '__main__':
    unittest.main()