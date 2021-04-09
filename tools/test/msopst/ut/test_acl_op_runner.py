import unittest
import pytest
from unittest import mock
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.st_report import OpSTReport
from op_test_frame.st.interface.acl_op_runner import AclOpRunner

class Process():
    def __init__(self, return_code=1):
        self.returncode = return_code

    def poll(self):
        return "None"

class TestUtilsMethods(unittest.TestCase):
    def test_compile1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.exists', return_value=False):
                runner = AclOpRunner('/home', 'ddd', 'report')
                runner.acl_compile()
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_compile2(self):
        report = OpSTReport()
        process = Process()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch('os.path.exists', return_value=True), mock.patch('os.chdir'):
                with mock.patch('subprocess.Popen', return_value=process):
                    runner = AclOpRunner('/home', 'ddd', report)
                    runner.acl_compile()

    def test_compile3(self):
        report = OpSTReport()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner.AclOpRunner._execute_command'):
                with mock.patch('os.path.exists', return_value=True), mock.patch('os.chdir'):
                    runner = AclOpRunner('/home', 'ddd', report)
                    runner.acl_compile()

    def test_run1(self):
        report = OpSTReport()
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.exists', return_value=False):
                runner = AclOpRunner('/home', 'ddd', report)
                runner.run()
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_run2(self):
        report = OpSTReport()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner.AclOpRunner._execute_command'):
                with mock.patch('os.path.exists',
                                return_value=True), mock.patch('os.chdir'):
                    runner = AclOpRunner('/home', 'ddd', report)
                    runner.run()

    def test_run3(self):
        report = OpSTReport()
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner.AclOpRunner._execute_command'):
                with mock.patch('os.path.exists',
                                return_value=True), mock.patch('os.chdir'):
                    runner = AclOpRunner('/home', 'ddd', report)
                    runner.process()


if __name__ == '__main__':
    unittest.main()