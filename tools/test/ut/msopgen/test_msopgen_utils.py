import unittest
import pytest
from unittest import mock
from op_gen.interface import utils


class TestUtilsMethods(unittest.TestCase):

    def test_print_error_log(self):
        utils.print_error_log("test error log")

    def test_read_json_file(self):
        with pytest.raises(utils.MsOpGenException) as error:
            utils.read_json_file('home/json_read')
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_OPEN_FILE_ERROR)

    def test_check_name_valid(self):
        utils.check_name_valid("")
        utils.check_name_valid('***')

    def test_make_dirs(self):
        with pytest.raises(utils.MsOpGenException) as error:
            with mock.patch('os.path.isdir', return_value=False):
                with mock.patch('os.makedirs', side_effect=OSError):
                    utils.make_dirs('/home/test1')
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_MAKE_DIRS_ERROR)

    def test_read_file(self):
        with pytest.raises(utils.MsOpGenException) as error:
            utils.read_file("/home/test_read_file")
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_READ_FILE_ERROR)

    def test_write_json_file(self):
        with pytest.raises(utils.MsOpGenException) as error:
            with mock.patch('os.fdopen', side_effect=IOError):
                utils.write_json_file('/home/test1', "ok")
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_WRITE_FILE_ERROR)

    def test_check_path_valid1(self):
        with pytest.raises(utils.MsOpGenException) as error:
            utils.check_path_valid('', True)
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_INVALID_PATH_ERROR)

    def test_check_path_valid2(self):
        with pytest.raises(utils.MsOpGenException) as error:
            with mock.patch('os.path.exists', return_value=False):
                utils.check_path_valid('/home/result.txt', False)
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_INVALID_PATH_ERROR)

    def test_check_path_valid3(self):
        with pytest.raises(utils.MsOpGenException) as error:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=False):
                    utils.check_path_valid('/home/result', False)
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_INVALID_PATH_ERROR)

    def test_check_path_valid4(self):
        with pytest.raises(utils.MsOpGenException) as error:
            with mock.patch('os.path.exists', return_value=False):
                with mock.patch('os.makedirs', side_effect=OSError):
                    utils.check_path_valid('/home/result', True)
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_INVALID_PATH_ERROR)

    def test_check_path_valid5(self):
        with pytest.raises(utils.MsOpGenException) as error:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=[True, False]):
                    utils.check_path_valid('/home/result', True)
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_INVALID_PATH_ERROR)

    def test_check_path_valid6(self):
        with pytest.raises(utils.MsOpGenException) as error:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=True):
                    with mock.patch('os.path.isdir', return_value=False):
                        utils.check_path_valid('/home/result', True)
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_INVALID_PATH_ERROR)

    def test_check_path_valid7(self):
        with pytest.raises(utils.MsOpGenException) as error:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=True):
                    with mock.patch('os.path.isfile', return_value=False):
                        utils.check_path_valid('/home/result', False)
        self.assertEqual(error.value.args[0],
                         utils.MS_OP_GEN_INVALID_PATH_ERROR)

    def test_fix_name_lower_with_under(self):
        before_convert_list = ["Abc", "AbcDef", "ABCDef", "Abc2DEf", "Abc2DEF","ABC2dEF"]
        after_convert_list = ["abc", "abc_def", "abc_def", "abc2d_ef", "abc2def", "abc2d_ef"]
        result = []
        for i in before_convert_list:
            result.append(utils.fix_name_lower_with_under(i))
        self.assertEqual(result, after_convert_list)

if __name__ == '__main__':
    unittest.main()