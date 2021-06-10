import unittest
import pytest
import os
import sys
from unittest import mock
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.subcase_design import SubCaseDesign
from op_test_frame.st.interface.st_report import OpSTReport

report = OpSTReport()
case_design = SubCaseDesign("test.json", {"name": "add"}, 1,[], report)

class TestUtilsMethods(unittest.TestCase):
    def test_check_key_exist_error(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_key_exist({"name": "add"}, "type", "INPUT")
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_check_shape_valid_error1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_shape_valid(["a",1,2])
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_check_shape_valid_error2(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_shape_valid([-3,1,2])
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_check_range_value_valid_error1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_range_value_valid([-3,1,2])
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_check_range_value_valid_error2(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_range_value_valid(["-3",1])
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_check_range_value_valid_error3(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_range_value_valid([6,1])
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_check_bin_valid_error(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_bin_valid("error.py","/home")
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_check_name_type_valid_error1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_range_value_valid({"a":1},"a")
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_check_name_type_valid_error2(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_range_value_valid({"a":""},"a")
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def test_check_name_type_valid_error3(self):
        with pytest.raises(utils.OpTestGenException) as error:
            case_design._check_range_value_valid({"type":"error"},"type")
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_DATA_ERROR)