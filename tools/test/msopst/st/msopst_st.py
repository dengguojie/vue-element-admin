import json
import sys
import unittest
import pytest
from unittest import mock
import os
import importlib.util
import importlib.machinery
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.data_generator import DataGenerator
from util import test_utils

sys.path.append(os.path.dirname(__file__) + "/../../")

loader = importlib.machinery.SourceFileLoader('msopst',
                                              os.path.dirname(__file__) +
                                              "/../../../op_test_frame/python"
                                              "/op_test_frame/scripts/msopst")
spec = importlib.util.spec_from_loader(loader.name, loader)
msopst = importlib.util.module_from_spec(spec)
loader.exec_module(msopst)

ST_GOLDEN_OUTPUT = './msopst/golden/base_case/golden_output/tbe'
ST_OUTPUT = './msopst/golden/base_case/output/'
INI_INPUT = './msopst/golden/base_case/input/conv2_d.ini'

# mindspore params info
ST_MS_GOLDEN_JSON_OUTPUT = './msopst/golden/base_case/golden_output' \
                           '/mindspore/json'

# ORI_FORMAT/ORI_SHAPE OUTPUT
ST_ORI_GOLDEN_OUTPUT = './msopst/golden/base_case/golden_output/tbe/ori_proj'

# AICPU_PARSE_HEAD_FILE OUTPUT
AICPU_PROJECT_INI_INPUT = './msopst/golden/base_case/golden_output/aicpu' \
                          '/cpukernel/op_info_cfg/aicpu_kernel/bucketize.ini'
AICPU_CASE_JSON_GOLDEN_OUTPUT = './msopst/golden/base_case/golden_output' \
                                '/aicpu/json'


class NumpyArrar:
    def tofile(self, file_path):
        pass


class TestUtilsMethods(unittest.TestCase):

    def test_create_case_json_from_ini_compare_success(self):
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'create', '-i', INI_INPUT, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopst.main()
        self.assertTrue(test_utils.check_file_context(ST_OUTPUT,
                                                      ST_GOLDEN_OUTPUT))

    @staticmethod
    def mock_gen_data(cls, *args, **kwargs):
        return NumpyArrar()

    # -------AR:TBE gen acl_op code-----------------------------------------
    def test_run_acl_op_from_case_json_compare_success(self):
        """run cmd"""
        self.gen_data_ = DataGenerator.gen_data
        DataGenerator.gen_data = staticmethod(self.mock_gen_data)
        data = json.dumps(
            [{'op': 'aa',
              'input_desc': [
                   {'format': ['ND'], 'type': ['int16'],
                    'shape': [[1, 3]], 'data_distribute': ['relu', 'beta']}],
              'output_desc': [{'format': ['ND'], 'type': ['int16'],
                               'shape': [[1, 3]]}],
              'attr': [{'name': 'ee', 'type': 'list_list_int',
                        'value': [[1, 2]]}],
              'case_name': 'xxx'}])
        args = ['aaa.py', 'run', '-i', '/home/a.json', '-soc', 'ascend310']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.acl_op_generator'
                        '.copy_template'):
                    with mock.patch(
                            'op_test_frame.st.interface.acl_op_generator'
                            '._write_content_to_file'):
                        with mock.patch('op_test_frame.st.interface.'
                                        'acl_op_runner.AclOpRunner.process'):
                            with mock.patch(
                                    'op_test_frame.st.interface.utils'
                                    '.check_path_valid'):
                                with mock.patch('builtins.open',
                                     mock.mock_open(read_data=data)) \
                                        as open_file:
                                    with mock.patch('os.makedirs'):
                                        with mock.patch('os.chmod'):
                                            open_file.write = None
                                            msopst.main()
        self.assertEqual(
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)
        DataGenerator.gen_data = self.gen_data_

    # --------------AR: msopst support for mindspore operator ----------------
    @unittest.mock.patch('op_test_frame.st.interface.case_generator.getattr')
    def test_create_case_json_from_mindspore_py_compare_success(self,
                                                                getattr_mock):
        """create cmd"""
        test_utils.clear_out_path(ST_OUTPUT)
        op_info = {'op_name': 'Square',
                   'inputs': [{'index': 0, 'name': 'x', 'need_compile': False,
                               'param_type': 'required', 'shape': 'all'}],
                   'outputs': [{'index': 0, 'name': 'y',
                                'need_compile': False,
                                'param_type': 'required',
                                'shape': 'all'}],
                   'attr': [],
                   'fusion_type': 'OPAQUE',
                   'dtype_format': [(('float32', 'DefaultFormat'),
                                    ('float32', 'DefaultFormat'))],
                   'imply_type': 'TBE', 'async_flag': False,
                   'binfile_name': 'square.so', 'compute_cost': 10,
                   'kernel_name': 'square_impl', 'partial_flag': 'True',
                   'reshape_type': '', 'dynamic_format': False,
                   'dynamic_shape': False, 'op_pattern': ''}

        getattr_mock.return_value = op_info
        lines = ''
        args = ['aaa.py', 'create', '-i', 'op_impl.py', '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch('op_test_frame.st.interface.utils'
                                '.check_path_valid'):
                    with mock.patch('builtins.open', mock.mock_open(
                            read_data=lines)):
                        with mock.patch('op_test_frame.st.interface'
                                        '.case_generator'
                                        '.importlib.import_module'):
                            msopst.main()

        self.assertTrue(test_utils.check_file_context(
            ST_OUTPUT, ST_MS_GOLDEN_JSON_OUTPUT))

    def test_run_case_json_from_mindspore_py_compare_success(self):
        self.gen_data_ = DataGenerator.gen_data
        DataGenerator.gen_data = staticmethod(self.mock_gen_data)
        test_utils.clear_out_path(ST_OUTPUT)
        data = json.dumps([{"case_name": "Test_Add_001",
                            "st_mode": "ms_python_train",
                            "op": "Add",
                            "input_desc": [
                                 {"format": ["NC1HWC0"],
                                  "ori_format": ["NCHW"],
                                  "type": "float16",
                                  "shape": [8, 1, 16, 4, 16],
                                  "ori_shape": [8, 1, 16, 4, 16],
                                  "data_distribute": ["uniform"],
                                  "value_range": [[0.1, 1.0]]},
                                 {"format": ["NC1HWC0"],
                                  "ori_format": ["NCHW"],
                                  "type": "float16",
                                  "shape": [8, 1, 16, 4, 16],
                                  "ori_shape": [8, 16, 16, 4],
                                  "data_distribute": ["uniform"],
                                  "value_range": [[0.1, 1.0]]}],
                            "output_desc": [{
                                 "format": ["NC1HWC0"],
                                 "ori_format": ["NCHW"],
                                 "type": "float16",
                                 "shape": [8, 1, 16, 4, 16],
                                 "ori_shape": [8, 16, 16, 4]}]
                            }])
        args = ['aaa.py', 'run', '-i', '/home/a.json', '-soc', 'ascend910',
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('op_test_frame.st.interface.'
                                'ms_op_runner.MsOpRunner.process'):
                    with mock.patch(
                            'op_test_frame.st.interface.utils'
                            '.check_path_valid'):
                        with mock.patch('builtins.open', mock.mock_open(
                                read_data=data)) as open_file:
                            with mock.patch('os.makedirs'):
                                with mock.patch('os.chmod'):
                                    open_file.write = None
                                    msopst.main()
        self.assertEqual(
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)
        DataGenerator.gen_data = self.gen_data_

    # # # ---------------AR: ori_format/ori_shape--------------------------
    def test_run_case_json_of_ori_format_compare_success(self):
        test_utils.clear_out_path(ST_OUTPUT)
        self.gen_data_ = DataGenerator.gen_data
        DataGenerator.gen_data = staticmethod(self.mock_gen_data)
        data = json.dumps([{"case_name": "Test_Add_001",
                            "op": "Add",
                            "st_mode": '',
                            "input_desc": [
                                 {"format": ["NC1HWC0"],
                                  "ori_format": ["NCHW"],
                                  "type": "float16",
                                  "shape": [8, 1, 16, 4, 16],
                                  "ori_shape": [8, 1, 16, 4, 16],
                                  "data_distribute": ["uniform"],
                                  "value_range": [[0.1, 1.0]]},
                                 {"format": ["NC1HWC0"],
                                  "ori_format": ["NCHW"],
                                  "type": "float16",
                                  "shape": [8, 1, 16, 4, 16],
                                  "ori_shape": [8, 16, 16, 4],
                                  "data_distribute": ["uniform"],
                                  "value_range": [[0.1, 1.0]]}],
                            "output_desc": [{
                                 "format": ["NC1HWC0"],
                                 "ori_format": ["NCHW"],
                                 "type": "float16",
                                 "shape": [8, 1, 16, 4, 16],
                                 "ori_shape": [8, 16, 16, 4]}]
                            }])

        args = ['aaa.py', 'run', '-i', '/home/a.json', '-soc', 'ascend310',
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.acl_op_generator'
                        '.copy_template'):
                    with mock.patch(
                            'op_test_frame.st.interface.acl_op_generator'
                            '._write_content_to_file'):
                        with mock.patch('op_test_frame.st.interface.'
                                        'acl_op_runner.AclOpRunner.process'):
                            with mock.patch(
                                    'op_test_frame.st.interface.utils'
                                    '.check_path_valid'):
                                with mock.patch('builtins.open',
                                     mock.mock_open(read_data=data)) \
                                        as open_file:
                                    with mock.patch('os.makedirs'):
                                        with mock.patch('os.chmod'):
                                            open_file.write = None
                                            msopst.main()
        self.assertEqual(
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)
        DataGenerator.gen_data = self.gen_data_

    # -----------------------AR: device_id--------------------------
    def test_run_case_json_of_device_id_compare_success(self):
        """run cmd"""
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', 'case.json', '-out', ST_OUTPUT, '-soc',
                'ascend310', '-d', 'XXX']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('op_test_frame.st.interface.utils.'
                                'check_path_valid'):
                    with mock.patch('builtins.open', mock.mock_open(
                            read_data='')):
                        msopst.main()
        self.assertEqual(error.value.code,
                         utils.OP_TEST_GEN_INVALID_DEVICE_ID_ERROR)

    # -------AR: compare expect data and execute data---------------
    def test_run_case_json_of_obtain_expect_data_compare_success(self):
        """run cmd"""
        test_utils.clear_out_path(ST_OUTPUT)
        self.gen_data_ = DataGenerator.gen_data
        DataGenerator.gen_data = staticmethod(self.mock_gen_data)
        data = json.dumps([{"case_name": "Test_Add_001",
                            "op": "Add",
                            "st_mode": '',
                            "calc_expect_func_file": "/home/aa.py"
                                                     ":calc_expect_func",
                            "input_desc": [
                                 {"format": ["NC1HWC0"],
                                  "ori_format": ["NCHW"],
                                  "type": "float16",
                                  "shape": [8, 1, 16, 4, 16],
                                  "ori_shape": [8, 1, 16, 4, 16],
                                  "data_distribute": ["uniform"],
                                  "value_range": [[0.1, 1.0]]}],
                            "output_desc": [{
                                 "format": ["NC1HWC0"],
                                 "ori_format": ["NCHW"],
                                 "type": "float16",
                                 "shape": [8, 1, 16, 4, 16],
                                 "ori_shape": [8, 16, 16, 4]}]
                            }])
        args = ['msopst', 'run', '-i', 'case.json', '-soc', 'ascend310',
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.acl_op_generator'
                        '.copy_template'):
                    with mock.patch(
                            'op_test_frame.st.interface.acl_op_generator'
                            '._write_content_to_file'):
                        with mock.patch('op_test_frame.st.interface'
                                        '.case_generator'
                                        '.importlib.import_module'):
                            with mock.patch('op_test_frame.st.interface'
                                            '.acl_op_runner.AclOpRunner'
                                            '.process'):
                                with mock.patch(
                                        'op_test_frame.st.interface.utils'
                                        '.check_path_valid'):
                                    with mock.patch('builtins.open',
                                         mock.mock_open(read_data=data)) \
                                            as open_file:
                                        with mock.patch('os.makedirs'):
                                            with mock.patch('os.chmod'):
                                                open_file.write = None
                                                msopst.main()
        self.assertEqual(
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)
        DataGenerator.gen_data = self.gen_data_

    # -------AR: aicpu support parse head file to crate case.json---------
    def test_run_case_json_of_aicpu_parse_head_file_compare_success(self):
        test_utils.clear_out_path(ST_OUTPUT)

        args = ['aaa.py', 'create', '-i', AICPU_PROJECT_INI_INPUT,
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopst.main()
        self.assertTrue(test_utils.check_file_context(
            ST_OUTPUT, AICPU_CASE_JSON_GOLDEN_OUTPUT))


if __name__ == '__main__':
    unittest.main()
