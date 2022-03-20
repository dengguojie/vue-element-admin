import unittest
import pytest
from unittest import mock
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager
from op_test_frame.st.interface import acl_op_generator
from op_test_frame.st.interface.atc_transform_om import AtcTransformOm
from op_test_frame.st.interface.st_report import OpSTReport

CASE_LIST = [{'op': 'Op'}]


class TestUtilsMethods(unittest.TestCase):
    def test_msopst_write_content_to_file_error(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.fdopen', side_effect=OSError):
                AtcTransformOm._write_content_to_file("content", "/home")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)

    def test_msopst_create_acl_op_json_content_error(self):
        report = OpSTReport()
        with mock.patch('json.dumps', side_effect=TypeError):
            with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
                with mock.patch('os.makedirs', return_value=True):
                    atc_transform = AtcTransformOm(CASE_LIST, '/home', None, False, report)
                    atc_transform.create_acl_op_json_content("", "", None)

    def test_msopst_append_content_to_file_error(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('builtins.open', side_effect=OSError):
                acl_op_generator._append_content_to_file("content", "/home")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)

    def test_msopst_copy_template_error1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.isdir', return_value=True):
                with mock.patch('os.listdir', return_value=["a","b","c"]):
                    acl_op_generator.copy_template("/home", "/home")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_MAKE_DIR_ERROR)

    def test_msopst_copy_template_error2(self):
        with pytest.raises(OSError) as error:
            with mock.patch('os.path.isdir', side_effect=OSError):
                with mock.patch('os.listdir', return_value=["a","b","c"]):
                    acl_op_generator.copy_template("/home", "/home")

if __name__ == '__main__':
    unittest.main()
