import unittest
import pytest
from unittest import mock
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager


class TestUtilsMethods(unittest.TestCase):

    def test_msopst_print_error_log(self):
        utils.print_error_log("test error log")

    def test_msopst_create_attr_value_str_None(self):
        utils.create_attr_value_str(None)
        utils.create_attr_value_str(['a'])
        utils.create_attr_value_str([True, 0])
        utils.create_attr_value_str(True)
        utils.create_attr_value_str('a')
        utils.create_attr_value_str([])

    def test_msopst_format_dict_to_list(self):
        result = utils.format_dict_to_list("{1,64}")
        self.assertEqual(result,"[1,64]")

    def test_msopst_check_path_valid1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            utils.check_path_valid("")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)

    def test_msopst_check_path_valid2(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.makedirs', side_effect=OSError):
                utils.check_path_valid("/test", True)
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_check_path_valid3(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=False):
                    utils.check_path_valid("/test")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_check_path_valid4(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=False):
                    utils.check_path_valid("/test", True)
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_check_path_valid5(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=True):
                    with mock.patch('os.path.isdir', return_value=False):
                        utils.check_path_valid("/test", True)
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_check_path_valid6(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=True):
                    with mock.patch('os.path.isdir', return_value=False):
                        utils.check_path_valid("/test")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_get_content_from_double_quotes(self):
        with pytest.raises(SystemExit) as error:
            utils.get_content_from_double_quotes("test")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_CONFIG_OP_DEFINE_ERROR)

    def test_msopst_check_value_valid(self):
        utils.check_value_valid("string", "value", "name")
        with pytest.raises(utils.OpTestGenException) as error:
            utils.check_value_valid("float", "value", "name")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        with pytest.raises(utils.OpTestGenException) as error:
            utils.check_value_valid("list_int", "value", "name")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        with pytest.raises(utils.OpTestGenException) as error:
            utils.check_value_valid("list_int", [], "name")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_msopst_check_attr_value_valid(self):
        attr = {'type': 'list_int', 'value': 'not_list', 'name': 'xx'}
        with pytest.raises(utils.OpTestGenException) as error:
            utils.check_attr_value_valid(attr)
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_msopst_load_json_file1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('builtins.open', side_effect=IOError):
                utils.load_json_file('/home/result')
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_OPEN_FILE_ERROR)

    def test_msopst_load_json_file2(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('builtins.open', mock.mock_open(read_data=b'[{')):
                utils.load_json_file('/home/result')
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_PARSE_JSON_FILE_ERROR)

    def test_msopst_read_file1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('builtins.open', side_effect=IOError):
                utils.read_file('/home/result')
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_OPEN_FILE_ERROR)

    def test_msopst_write_json_file1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.fdopen', side_effect=IOError):
                utils.write_json_file('/home/result', "test")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)

    def test_msopst_make_dirs(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.makedirs', side_effect=OSError):
                utils.make_dirs('/home/result')
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_MAKE_DIRS_ERROR)

    def test_fix_name_lower_with_under(self):
        before_convert_list = ["Abc", "AbcDef", "ABCDef", "Abc2DEf", "Abc2DEF","ABC2dEF"]
        after_convert_list = ["abc", "abc_def", "abc_def", "abc2d_ef", "abc2def", "abc2d_ef"]
        result = []
        for i in before_convert_list:
            result.append(utils.fix_name_lower_with_under(i))
        self.assertEqual(result, after_convert_list)

if __name__ == '__main__':
    unittest.main()