import unittest
import pytest
from unittest import mock
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager
from op_test_frame.st.interface import model_parser

class Args:
    def __init__(self, input_file, output_path, model_path):
        self.input_file = input_file
        self.output_path = output_path
        self.model_path = model_path
        self.quiet = False

class TestUtilsMethods(unittest.TestCase):
    def test_msopst_get_model_nodes1(self):
        args = Args('./a.json', '', './a.pbv')
        with pytest.raises(utils.OpTestGenException) as error:
            model_parser.get_model_nodes(args,'')
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)

    def test_msopst_get_model_nodes2(self):
        args = Args('./a.json', '', './a.pb')
        with pytest.raises(utils.OpTestGenException) as error:
            model_parser.get_model_nodes(args,'')
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)

if __name__ == '__main__':
    unittest.main()