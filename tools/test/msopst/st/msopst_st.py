import sys
import unittest
import pytest
from unittest import mock
import os
import importlib.util
import importlib.machinery
sys.path.append(os.path.dirname(__file__)+"/../../")
from util import test_utils

loader = importlib.machinery.SourceFileLoader('msopst', os.path.dirname(__file__)+
                                              "/../../../op_test_frame/python/op_test_frame/scripts/msopst")
spec = importlib.util.spec_from_loader(loader.name, loader)
msopst = importlib.util.module_from_spec(spec)
loader.exec_module(msopst)

ST_GOLDEN_OUTPUT = './msopst/golden/base_case/golden_output'
ST_OUTPUT = './msopst/golden/base_case/output/'
INI_INPUT = './msopst/golden/base_case/input/conv2_d.ini'


class TestUtilsMethods(unittest.TestCase):

    def test_create_case_json_from_ini_compare_success(self):
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'create', '-i', INI_INPUT, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopst.main()
        self.assertTrue(test_utils.check_file_context(ST_OUTPUT,
                                                      ST_GOLDEN_OUTPUT))


if __name__ == '__main__':
    unittest.main()