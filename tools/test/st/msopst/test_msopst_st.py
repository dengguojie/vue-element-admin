import json
import sys
import unittest
import filecmp

import numpy as np
import pytest
from unittest import mock
import os
import importlib.util
import importlib.machinery
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.arg_parser import MsopstArgParser
from op_test_frame.st.interface.const_manager import ConstManager
from op_test_frame.st.interface import model_parser
from op_test_frame.st.interface.framework import tf_model_parser
from op_test_frame.st.interface.framework.tf_model_parser import TFModelParse
from op_test_frame.st.interface import result_comparer
from op_test_frame.st.interface.case_generator import CaseGenerator
from op_test_frame.st.interface.data_generator import DataGenerator
from op_test_frame.st.interface.st_report import OpSTReport
from op_test_frame.st.interface.st_report import OpSTCaseReport
from op_test_frame.st.interface.st_report import ReportJsonEncoder
from op_test_frame.st.interface.op_st_case_info import OpSTCase
from op_test_frame.st.interface.op_st_case_info import OpSTCaseTrace
from op_test_frame.st.interface.acl_op_runner import AclOpRunner
from op_test_frame.st.interface.atc_transform_om import AtcTransformOm
from op_test_frame.st.interface import acl_op_runner
from op_test_frame.st.interface import ms_op_generator
from util import test_utils
import test_pytorch_model_parser

sys.path.append(os.path.dirname(__file__) + "/../../")

loader = importlib.machinery.SourceFileLoader('msopst',
                                              os.path.dirname(__file__) +
                                              "/../../../op_test_frame/python"
                                              "/op_test_frame/scripts/msopst")
spec = importlib.util.spec_from_loader(loader.name, loader)
msopst = importlib.util.module_from_spec(spec)
loader.exec_module(msopst)

# TBE OPERATOR INPUT/OUTPOUT
ST_GOLDEN_OUTPUT = './st/msopst/golden/base_case/golden_output/tbe'
ST_OUTPUT = './st/msopst/golden/base_case/output/'
INI_INPUT = './st/msopst/golden/base_case/input/conv2_d.ini'
MODEL_ARGS = './st/msopst/golden/base_case/input/add.pb'
ST_GOLDEN_OP_RESULT_TXT = './st/msopst/golden/base_case/input' \
                          '/result.txt'
ST_GOLDEN_ABNORMAL_CASE_JSON_INPUT = './st/msopst/golden/base_case/input'\
                                     '/test_add_abnormal_case.json'

# AICPU_PARSE_HEAD_FILE OUTPUT
BUCKETIZE_INI_INPUT = './st/msopst/golden/base_case/golden_output/aicpu' \
                          '/cpukernel/op_info_cfg/aicpu_kernel/bucketize.ini'
TOPK_INI_INPUT = './st/msopst/golden/base_case/golden_output/aicpu' \
                          '/cpukernel/op_info_cfg/aicpu_kernel/top_k.ini'
LESS_INI_INPUT = './st/msopst/golden/base_case/golden_output/aicpu' \
                          '/cpukernel/op_info_cfg/aicpu_kernel/less.ini'
CAST_INI_INPUT = './st/msopst/golden/base_case/golden_output/aicpu' \
                          '/cpukernel/op_info_cfg/aicpu_kernel/cast.ini'
AICPU_CASE_JSON_GOLDEN_OUTPUT = './st/msopst/golden/base_case/golden_output' \
                                '/aicpu/json'

# paramType: optional OUTPUT
OPTIONAL_INI_INPUT = './st/msopst/golden/base_case/input/Pooling.ini'
OPTIONAL_ST_GOLDEN_OUTPUT = './st/msopst/golden/base_case/' \
                            'golden_output/optional_input'
ST_GOLDEN_OP_CASE_JSON_INPUT = './st/msopst/golden/base_case/input' \
                               '/Pooling_case_20210225145706.json'
ST_GOLDEN_ACL_PROJECT_OUTPUT_TESTCASE = './st/msopst/golden/base_case/golden_output' \
                                   '/gen_optional_acl_prj/Pooling/src/testcase.cpp'
ST_GOLDEN_ACL_PROJECT_OUTPUT_RUN = './st/msopst/golden/base_case/golden_output' \
                                   '/gen_optional_acl_prj/Pooling/run/out'\
                                   '/test_data/config/'
MSOPST_CONF_INI = './st/msopst/golden/base_case/input/msopst.ini'

# dynamic shape
ST_GOLDEN_OP_DYNAMIC_SHAPE_INI_INPUT = './st/msopst/golden/base_case/input' \
                                       '/add.ini'

ST_GOLDEN_OP_DYNAMIC_SHAPE_JSON_INPUT = './st/msopst/golden/base_case/input' \
                                        '/add_case.json'
ST_GOLDEN_DYNAMIC_SHAPE_TESTCASE = './st/msopst/golden/base_case/golden_output/' \
                                   'gen_optional_acl_prj/Add/src/testcase.cpp'

# MINDSPORE OPERATOR INPUT/OUTPOUT
ST_MS_GOLDEN_JSON_OUTPUT = './st/msopst/golden/base_case/golden_output' \
                           '/mindspore/json'
ST_MS_GOLDEN_INPUT_JSON = './st/msopst/golden/base_case/input/ms_st_report.json'

ST_GOLDEN_OP_GEN_WITH_VALUE_JSON_INPUT = './st/msopst/golden/base_case/input' \
                                             '/test_value_add.json'
ST_GOLDEN_OP_GEN_WITH_VALUE_ACL_PROJECT_OUTPUT = './st/msopst/golden/base_case' \
                                                 '/golden_output/' \
                                                 'gen_optional_acl_prj/Adds/src'
ST_MS_GOLDEN_INPUT_JSON_WITH_VALUE = './st/msopst/golden/base_case/input/ms_case_with_value.json'
ST_GOLDEN_OP_FUZZ_CASE_JSON_INPUT = './st/msopst/golden/base_case/input' \
                                    '/test_add_fuzz.json'
ST_GOLDEN_OP_FUZZ_CASE_OUTPUT_SRC = './st/msopst/golden/base_case/golden_output' \
                                   '/fuzz/Add/src/testcase.cpp'
ST_GOLDEN_OP_FUZZ_CASE_OUTPUT_RUN = './st/msopst/golden/base_case/golden_output' \
                                   '/fuzz/Add/run/out'\
                                   '/test_data/config/'
ST_GOLDEN_MS_FUZZ_CASE_JSON_INPUT = './st/msopst/golden/base_case/input' \
                                    '/ms_case_fuzz.json'
ST_GOLDEN_MS_FUZZ_CASE_OUTPUT_SRC = './st/msopst/golden/base_case/golden_output' \
                                   '/fuzz/Square/src'
# const input
ST_GOLDEN_CONST_INPUT_VALUE_JSON = './st/msopst/golden/base_case/input' \
                                    '/const_input_with_value.json'
ST_GOLDEN_CONST_INPUT_DATA_DISTRIBUTE_JSON = './st/msopst/golden/base_case/input' \
                                    '/const_input_with_data_distribute.json'
ST_GOLDEN_CONST_INPUT_NO_VALUE_JSON = './st/msopst/golden/base_case/input' \
                                    '/const_input_with_no_value.json'
ST_GOLDEN_SCALAR_INPUT_WITH_VALUE_JSON = './st/msopst/golden/base_case/input' \
                                    '/scalar_input_with_value.json'
# attr support data type
ST_GOLDEN_ATTR_SUPPORT_DATA_TYPE_JSON = './st/msopst/golden/base_case/input' \
                                    '/test_attr_support_data_type.json'
ST_GOLDEN_ATTR_SUPPORT_DATA_TYPE_TESTCASE = './st/msopst/golden/base_case/golden_output' \
                                        '/gen_optional_acl_prj/TestOp/testcase.cpp'
ST_GOLDEN_ATTR_SUPPORT_DATA_TYPE_ACL_JSON = './st/msopst/golden/base_case/golden_output' \
                                            '/gen_optional_acl_prj/TestOp/acl_op.json'
# const input golden files.
ST_GOLDEN_CONST_INPUT_SRC_TESTCASE = './st/msopst/golden/base_case/golden_output' \
                                        '/gen_optional_acl_prj/ResizeBilinearV2/src/testcase.cpp'
ST_GOLDEN_CONST_INPUT_SRC_OP_EXECUTE = './st/msopst/golden/base_case/golden_output' \
                                       '/gen_optional_acl_prj/ResizeBilinearV2/src/op_execute.cpp'
ST_GOLDEN_CONST_INPUT_CONFIG_ACL_OP = './st/msopst/golden/base_case/golden_output' \
                                        '/gen_optional_acl_prj/ResizeBilinearV2/config/acl_op.json'
ST_GOLDEN_SCALAR_INPUT_SRC_TESTCASE = './st/msopst/golden/base_case/golden_output' \
                                        '/gen_optional_acl_prj/const_input/TestScalar/testcase.cpp'
ST_GOLDEN_SCALAR_INPUT_CONFIG_ACL_OP = './st/msopst/golden/base_case/golden_output' \
                                        '/gen_optional_acl_prj/const_input/TestScalar/acl_op.json'

ST_GOLDEN_OP_ADD_REPORT_JSON_INPUT = './st/msopst/golden/base_case/input/add_st_report.json'
ST_GOLDEN_OP_POOLING_REPORT_JSON_INPUT = './st/msopst/golden/base_case/input/pooling_st_report.json'


class NumpyArrar:
    def tofile(self, file_path):
        pass


class Args:
    def __init__(self, input_file, output_path, model_path):
        self.input_file = input_file
        self.output_path = output_path
        self.model_path = model_path
        self.quiet = False


def compare_context(src_name, dst_name):
    if not filecmp.cmp(src_name, dst_name):
        print(" %s VS %s return false." % (src_name, dst_name))
        return False
    return True

class TestUtilsMethods(unittest.TestCase):

    def test_check_path_valid(self):
        args = ['aaa.py', 'create', '-i', '/home/a.ini']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.isdir', return_value=False):
                    msopst.main()
        self.assertEqual(error.value.code,
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    # ------------------create case.json for TBE ------------
    def test_create_cmd_for_tbe(self):
        """
        create case.json by .ini file, support for TBE operators.
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'create', '-i', INI_INPUT, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopst.main()
        self.assertTrue(test_utils.check_file_context(ST_OUTPUT,
                                                      ST_GOLDEN_OUTPUT))

    def test_create_cmd_for_aicpu(self):
        """
        create case.json by .ini file, support for AICPU operators.
        """
        test_utils.clear_out_path(ST_OUTPUT)

        args = ['msopst', 'create', '-i', CAST_INI_INPUT,
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils'
                        '.check_path_valid'):
                    msopst.main()
        self.assertEqual(
            error.value.code, ConstManager.OP_TEST_GEN_NONE_ERROR)

    # -----------------get information from model-----------------
    def test_create_cmd_and_update_aicore_info_from_model(self):
        """
        create case.json by .ini file and get information by models,
        support for ai_core operators.
        """
        test_utils.clear_out_path(ST_OUTPUT)
        data = [{'op_type': 'Add',
                 'layer': 'add',
                 'input_dtype': ['float', 'float'],
                 'output_dtype': ['float'],
                 'input_shape': [[1, 1, 5, 5], [1, 1, 5, 5]],
                 'output_shape': [[1, 1, 5, 5]],
                 'attr': [
                     {'name': 'T', 'type': 'type', 'value': ' DT_FLOAT'}]}]
        args = ['msopst', 'create', '-i', ST_GOLDEN_OP_DYNAMIC_SHAPE_INI_INPUT,
                '-out', ST_OUTPUT, '-m', MODEL_ARGS]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.model_parser.'
                        '_function_call', return_value=data):
                    msopst.main()
        self.assertEqual(
            error.value.code, ConstManager.OP_TEST_GEN_NONE_ERROR)

    def test_create_cmd_and_update_aicpu_info_from_model(self):
        """
        create case.json by .ini file and get information by models,
        support for aicpu operators.
        """
        test_utils.clear_out_path(ST_OUTPUT)
        data = [{'op_type': 'Add',
                 'layer': 'add',
                 'input_dtype': ['float', 'float'],
                 'output_dtype': ['float'],
                 'input_shape': [[1, 1, 5, 5], [1, 1, 5, 5]],
                 'output_shape': [[1, 1, 5, 5]],
                 'attr': [
                     {'name': 'T', 'type': 'type', 'value': ' DT_FLOAT'}]}]
        args = ['msopst', 'create', '-i', BUCKETIZE_INI_INPUT,
                '-out', ST_OUTPUT, '-m', MODEL_ARGS]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.model_parser.'
                        '_function_call', return_value=data):
                    msopst.main()
        self.assertEqual(
            error.value.code, ConstManager.OP_TEST_GEN_NONE_ERROR)

    def test_model_parser_1(self):
        """
        verify the abnormal scene of get_tf_model_nodes function
        in tf_model_parser.py
        """
        args = Args('./a.json', '', './a.pb')
        with pytest.raises(utils.OpTestGenException) as error:
            tf_model_parser = TFModelParse(args)
            tf_model_parser.get_tf_model_nodes("Conv2D")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_get_info_from_model_fun(self):
        """
        verify the normal scene of _get_info_from_model function
        in case_generator.py
        """
        args = Args('./a.json', '', '')
        case_gen = CaseGenerator(args)
        item = {'name': 'data_format', 'type': 's', 'value': '"NHWC"'}
        attr = {'name': 'data_format', 'type': 'string', 'value': 'NHWC'}
        attr_list = [{'name': 'data_format',
                      'type': 'string', 'value': 'NHWC'}]
        base_case = {'case_name': 'Test_XXX_sub_case_001', 'op': 'Conv2D',
                     'input_desc':
                         [
                             {'format': ['NHWC'],
                              'type': 'float',
                              'shape': [[1, 1, 1, 4096]],
                              'data_distribute': ['uniform'],
                              'value_range': [[0.1, 1.0]],
                              'name': 'x'}
                         ],
                     'output_desc': [
                         {
                             'format': ['NHWC'],
                             'type': 'float',
                             'shape': [[1, 1, 1, 1000]],
                             'name': 'y'}]}
        case_gen._get_info_from_model(item, attr, attr_list, base_case)

    def test_model_parser_2(self):
        """
        verify the scene of get_tf_model_node function in tf_model_parser.py
        """
        args = Args('./a.json', '', MODEL_ARGS)
        with mock.patch(
                'op_test_frame.st.interface.framework.'
                'tf_model_parser.input', return_value='n'):
            tf_model_parser = TFModelParse(args)
            tf_model_parser.get_tf_model_nodes("Conv2D")

    def test_model_parser_3(self):
        """
        verify the abnormal scene of change_shape function
        in tf_model_parser.py
        """
        args = Args('./a.json', '', MODEL_ARGS)
        lines = '{"Placeholder": ' \
                '{"ori_shape": [-1, 224, 224, 3], "new_shape": []}}'
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch(
                    'op_test_frame.st.interface.utils.check_path_valid'):
                with mock.patch('builtins.open',
                                mock.mock_open(read_data=lines)):
                    with mock.patch('os.open') as open_file, \
                            mock.patch('os.fdopen'):
                        open_file.write = None
                        tf_model_parser = TFModelParse(args)
                        tf_model_parser.change_shape()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_TF_CHANGE_PLACEHOLDER_ERROR)

    def test_model_parser_4(self):
        """
        verify the abnormal scene of get_model_nodes function
        in model_parser.py
        """
        args = Args('./a.json', '', './a.pbv')
        with pytest.raises(utils.OpTestGenException) as error:
            model_parser.get_model_nodes(args, '')
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)

    def test_model_parser_5(self):
        """
        verify the abnormal scene of _attr_value_shape_list function
        in tf_model_parser.py
        """
        attr_value_shape = [np.array([[1, 2], [3, 4]])]
        tf_model_parser._attr_value_shape_list(attr_value_shape)

    def test_model_parser_6(self):
        """
        verify the abnormal scene of _map_tf_input_output_dtype function
        in tf_model_parser.py
        """
        tf_dtype_list = ["float32", "float16", "int8", "int16", "int32",
                         "uint8", "uint16", "uint32", "bool", "int64"]
        for tf_dtype in tf_dtype_list:
            tf_model_parser._map_tf_input_output_dtype(tf_dtype)

    def test_model_parser_7(self):
        """
        verify the abnormal scene of get_shape function
        in tf_model_parser.py
        """
        args = Args('./a.json', '', './a.pb')
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch(
                    'op_test_frame.st.interface.utils.check_path_valid'):
                tf_model_parser = TFModelParse(args)
                tf_model_parser.get_shape()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_TF_LOAD_ERROR)

    # ---------------------compare with cpu data---------------------
    def test_run_cmd_and_compare_npu_data_and_cpu_data(self):
        """
        test compare npu data and cpu data
        """
        test_utils.clear_out_path(ST_OUTPUT)
        self.gen_data_ = DataGenerator.gen_data
        DataGenerator.gen_data = staticmethod(self.mock_gen_data)
        data = json.dumps([{"case_name": "Test_Add_001",
                            "op": "Add",
                            "st_mode": '',
                            "calc_expect_func_file": "/home/aa.py"
                                                     ":calc_expect_func",
                            "input_desc": [
                                {
                                    "format": ["NC1HWC0"],
                                    "ori_format": ["NCHW"],
                                    "type": "float16",
                                    "shape": [8, 1, 16, 4, 16],
                                    "ori_shape": [8, 1, 16, 4, 16],
                                    "data_distribute": ["uniform"],
                                    "value_range": [[0.1, 1.0]]
                                }],
                            "output_desc": [
                                {
                                    "format": ["NC1HWC0"],
                                    "ori_format": ["NCHW"],
                                    "type": "float16",
                                    "shape": [8, 1, 16, 4, 16],
                                    "ori_shape": [8, 16, 16, 4]
                                }]
                            }])
        args = ['msopst', 'mi', 'gen', '-i', 'case.json',
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.acl_op_generator'
                        '.copy_template'):
                    with mock.patch(
                            'op_test_frame.st.interface.atc_transform_om.AtcTransformOm'
                            '._write_content_to_file'):
                        with mock.patch('op_test_frame.st.interface'
                                        '.case_generator'
                                        '.importlib.import_module'):
                            with mock.patch(
                                    'op_test_frame.st.interface.utils'
                                    '.check_path_valid'):
                                with mock.patch(
                                        'builtins.open',
                                        mock.mock_open(read_data=data)) \
                                        as open_file:
                                    with mock.patch('os.open'):
                                        with mock.patch('os.fdopen'):
                                            with mock.patch('os.makedirs'):
                                                with mock.patch('os.chmod'):
                                                    open_file.write = None
                                                    msopst.main()
        self.assertEqual(
            error.value.code, ConstManager.OP_TEST_GEN_NONE_ERROR)
        DataGenerator.gen_data = self.gen_data_

    def test_compare_func_1(self):
        """
        verify the abnormal scene of compare function in result_comparer.py
        """
        report = OpSTReport()
        run_dir = "xxx.txt"
        err_thr = [0.01, 0.01]
        result_comparer.compare(report, run_dir, err_thr)

    def test_compare_func_2(self):
        """
        verify the normal scene of compare function in result_comparer.py
        """
        report = OpSTReport()
        op_st = OpSTCase("AddN", {"calc_expect_func_file_func": 1})
        op_st_case_trace = OpSTCaseTrace(op_st)
        op_st_report = OpSTCaseReport(op_st_case_trace)
        err_thr = [0.01, 0.01]
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.utils.execute_command'):
                with mock.patch('os.path.exists',
                                return_value=True), mock.patch('os.chdir'):
                    with mock.patch('os.access', return_value=True):
                        with mock.patch('os.path.join',
                                        return_value=ST_GOLDEN_OP_RESULT_TXT):
                            with mock.patch(
                                    'op_test_frame.st.interface.st_report.'
                                    'OpSTReport.get_case_report',
                                    return_value=op_st_report):
                                runner = AclOpRunner('/home', 'ddd', report)
                                runner.run()
                                result_comparer.compare(
                                    report, ST_GOLDEN_OP_RESULT_TXT, err_thr)

    def test_compare_func_3(self):
        """
        verify the normal scene of compare_by_path function in result_comparer.py
        """
        report = OpSTReport()
        op_st = OpSTCase("AddN", {"calc_expect_func_file_func": 1})
        op_st_case_trace = OpSTCaseTrace(op_st)
        op_st_report = OpSTCaseReport(op_st_case_trace)
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch('op_test_frame.st.interface.utils.execute_command'):
                with mock.patch('os.path.exists',
                                return_value=True), mock.patch('os.chdir'):
                    with mock.patch('os.access', return_value=True):
                        with mock.patch('os.listdir', return_value='AddN'):
                            with mock.patch(
                                    'os.path.join',
                                    return_value=ST_GOLDEN_OP_RESULT_TXT):
                                with mock.patch(
                                        'op_test_frame.st.interface.st_report.'
                                        'OpSTReport.get_case_report',
                                        return_value=op_st_report):
                                    runner = AclOpRunner(
                                        '/home', 'ddd', report)
                                    runner.run()
                                    result_comparer.compare_by_path(
                                        report, ST_GOLDEN_OP_RESULT_TXT)

    def test_compare_func_4(self):
        """
        verify the normal scene of data_compare function in result_comparer.py
        """
        err_thr = [0.01, 0.01]
        npu_output = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        cpu_output = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
        npu_output_array = np.array(npu_output)
        cpu_output_array = np.array(cpu_output)
        result_comparer._data_compare(npu_output_array, cpu_output_array, err_thr)

    def test_compare_func_5(self):
        """
        verify the normal scene of data_compare function in result_comparer.py
        """
        report = OpSTReport()
        op_st = OpSTCase("AddN", {"calc_expect_func_file_func": 1})
        op_st_case_trace = OpSTCaseTrace(op_st)
        op_st_report = OpSTCaseReport(op_st_case_trace)
        report.report_list = [op_st_report]
        report._summary_txt()

    # -----------------------case_generator--------------------------
    def test_case_generator_1(self):
        """
        verify the normal scene of _get_default_attr_value function
        in case_generator.py
        """
        args = Args('./a.json', '', '')
        case_gen = CaseGenerator(args)
        case_gen._get_default_attr_value("float", '0.1', 'xxx')
        case_gen._get_default_attr_value("int", '1', 'xxx')
        case_gen._get_default_attr_value("bool", 'true', 'xxx')
        case_gen._get_default_attr_value("str", 'stride', 'xxx')
        case_gen._get_default_attr_value("list", '[1]', 'xxx')
        case_gen._get_default_attr_value("listListInt", '[[1]]', 'xxx')
        case_gen._get_default_attr_value("listInt", '[1]', 'xxx')
        case_gen._get_default_attr_value("listFloat", '[0.1]', 'xxx')
        case_gen._get_default_attr_value("listStr", '[\'x\']', 'xxx')
        case_gen._get_default_attr_value("listBool", '[\'true\']', 'xxx')

    # -----------------------data_generator--------------------------
    def test_gen_data_1(self):
        """
        verify the normal scene of gen_data function
        in data_generator.py
        """
        report = OpSTReport()
        with pytest.raises(utils.OpTestGenException) as error:
            data_generator = DataGenerator([], '/home', True, report)

            data_generator.gen_data([1, 4], -10, 4, 'int32', 'xx')
            self.assertEqual(error.value.args[0],
                             ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)

    def test_gen_data_2(self):
        """
        verify the normal scene of gen_data function
        in data_generator.py
        """
        report = OpSTReport()
        data_generator = DataGenerator([], '/home', True, report)
        data_distribution = ['uniform', 'normal', 'beta', 'laplace',
                             'triangular', 'sigmoid', 'softmax', 'tanh']
        for distribution in data_distribution:
            data = data_generator.gen_data(
                [2, 4], -10, 400, 'float', distribution)
        self.assertEqual(2, len(data.shape))

    # ---------------------profiling_analysis-----------------------
    def test_profiling_analysis_1(self):
        """
        verify the normal scene of _get_op_case_result_and_show_data function
        in acl_op_runner.py
        """
        csv_file = "./st/msopst/golden/base_case/input/op_summary_0_1.csv"
        op_name_list = ["Cast", "Cast", "Cast", "Cast"]
        report = OpSTReport()
        runner = AclOpRunner('/home', 'ddd', report)
        runner._get_op_case_result_and_show_data(
            csv_file, op_name_list)

    def test_profiling_op_name_list_fail(self):
        """
        verify the normal scene of _get_op_case_result_and_show_data function
        in acl_op_runner.py
        """
        csv_file = "./st/msopst/golden/base_case/input/op_summary_0_1.csv"
        op_name_list = []
        report = OpSTReport()
        runner = AclOpRunner('/home', 'ddd', report)
        runner._prof_get_op_case_info_from_csv_file(
            csv_file, op_name_list)

    def test_profiling_csv_file_fail(self):
        """
        verify the normal scene of _get_op_case_result_and_show_data function
        in acl_op_runner.py
        """
        csv_file = "./st/msopst/golden/base_case/input/op_summary.csv"
        op_name_list = ["Cast", "Cast", "Cast", "Cast"]
        report = OpSTReport()
        runner = AclOpRunner('/home', 'ddd', report)
        runner._prof_get_op_case_info_from_csv_file(
            csv_file, op_name_list)

    def test_profiling_analysis_2(self):
        """
        verify the normal scene of _prof_get_op_name_from_report function
        in acl_op_runner.py
        """
        report = OpSTReport()
        op_st = OpSTCase("AddN", {"calc_expect_func_file_func": 1})
        op_st_case_trace = OpSTCaseTrace(op_st)
        op_st_report = OpSTCaseReport(op_st_case_trace)
        with mock.patch(
                'op_test_frame.st.interface.st_report.'
                'OpSTReport.get_case_report', return_value=op_st_report):
            runner = AclOpRunner('/home', 'ddd', report)
            run_result_list = ["1  Test_AddN_001_case_001  [pass]"]
            runner._prof_get_op_name_from_report(run_result_list)

    def test_profiling_show_data(self):
        each_case_info_list = ['Cast', 'AI_CPU', '1338.541672']
        op_case_info_list = []
        for i in range(30):
            op_case_info_list.append(each_case_info_list)
        acl_op_runner.display_op_case_info(op_case_info_list)

    def test_prof_analyze(self):
        out_path = "./st/msopst/golden/base_case/input"
        lines = "1  Test_AddN_001_case_001  [pass] \n " \
                "2  Test_AddN_001_case_002  [pass] \n " \
                "3  Test_AddN_001_case_003  [pass] \n"
        report = OpSTReport()
        runner = AclOpRunner('/home', 'ddd', report)
        with mock.patch(
                'op_test_frame.st.interface.utils.ScanFile.scan_subdirs',
                return_value=['result.txt']):
            with mock.patch(
                    'op_test_frame.st.interface.utils.read_file',
                    return_value=lines):
                with mock.patch('os.path.join', return_value=out_path), \
                     mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.chdir'):
                    with mock.patch('op_test_frame.st.interface.utils.execute_command'):
                        runner.prof_analyze(os.path.join(out_path, ConstManager.PROF))

    def test_prof_run_no_install_path(self):
        out_path = "./st/msopst/golden/base_case/input"
        report = OpSTReport()
        runner = AclOpRunner('/home', 'ddd', report)
        runner.prof_run(out_path)

    def test_prof_run(self):
        out_path = "./st/msopst/golden/base_case/input"
        lines = "1  Test_AddN_001_case_001  [pass] \n " \
                "2  Test_AddN_001_case_002  [pass] \n " \
                "3  Test_AddN_001_case_003  [pass] \n"
        report = OpSTReport()
        runner = AclOpRunner('/home', 'ddd', report)
        with mock.patch('os.getenv', return_value="/home/test/Ascend"):
            with mock.patch(
                    'op_test_frame.st.interface.utils.ScanFile.scan_subdirs',
                    return_value=['result.txt']):
                with mock.patch(
                        'op_test_frame.st.interface.utils.read_file',
                        return_value=lines):
                    with mock.patch('os.path.join', return_value=out_path), \
                         mock.patch('os.path.exists', return_value=True), \
                         mock.patch('os.access', return_value=True), \
                         mock.patch('os.chdir'):
                        with mock.patch(
                                'op_test_frame.st.interface.utils.execute_command'):
                            runner.prof_run(out_path)

    def test_scan_subdirs(self):
        prof_base_path = '/home/test'
        scan = utils.ScanFile(prof_base_path, first_prefix="PROF",
                              second_prefix="device")
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.path.isdir', return_value=True):
                with mock.patch('os.path.split', return_value=["PROF_A", "device_0"]):
                    with mock.patch('os.listdir',
                                    return_value=["PROF_A/device_0"]):
                        scan.scan_subdirs()

    def test_run_st_attr_data_type(self):
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i',
                ST_GOLDEN_ATTR_SUPPORT_DATA_TYPE_JSON, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()
        testcase_cpp = os.path.join(ST_OUTPUT,
                                          'TestOp/src/testcase.cpp')
        acl_op_json = os.path.join(ST_OUTPUT, 'TestOp/run/out/test_data'
                                         '/config/acl_op.json')
        self.assertTrue(compare_context(
            testcase_cpp, ST_GOLDEN_ATTR_SUPPORT_DATA_TYPE_TESTCASE))
        self.assertTrue(compare_context(
            acl_op_json, ST_GOLDEN_ATTR_SUPPORT_DATA_TYPE_ACL_JSON))

    def test_st_report_save(self):
        """
        test_st_report_save
        """
        test_utils.clear_out_path(ST_OUTPUT)
        report = OpSTReport()
        report_data_path = os.path.join(ST_OUTPUT, 'st_report.json')
        with mock.patch(
                'op_test_frame.st.interface.st_report.OpSTReport._to_json_obj',
                return_value=[{}]):
            report.save(report_data_path)
        with mock.patch(
                'op_test_frame.st.interface.st_report.OpSTReport._to_json_obj',
                return_value=[{}]):
            report.save(report_data_path)

    # --------------ori_format/ori_shape/device_id--------------------
    def test_gen_ori_format_or_shape_src_code(self):
        """
        test generate acl code for ori_format/ori_shape/device_id
        of TBE operators
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'mi', 'gen_testcase', '-i', ST_GOLDEN_OP_POOLING_REPORT_JSON_INPUT,
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils'
                        '.check_path_valid'):
                    msopst.main()
        op_output_testcase = os.path.join(ST_OUTPUT, 'src/testcase.cpp')
        self.assertTrue(compare_context(
            op_output_testcase, ST_GOLDEN_ACL_PROJECT_OUTPUT_TESTCASE))

    # ------------------------parse aicpu head file----------------
    def test_create_cmd_for_aicpu_parse_head_file(self):
        """
        test create cmd of aicpu support parse head file to create case.json
        """
        test_utils.clear_out_path(ST_OUTPUT)

        args = ['msopst', 'create', '-i', BUCKETIZE_INI_INPUT,
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopst.main()
        self.assertTrue(test_utils.check_file_context(
            ST_OUTPUT, AICPU_CASE_JSON_GOLDEN_OUTPUT))

    def test_create_cmd_for_aicpu_parse_head_file_with_attr(self):
        """
        test create cmd of aicpu support parse head file to create case.json
        """
        test_utils.clear_out_path(ST_OUTPUT)

        args = ['msopst', 'create', '-i', TOPK_INI_INPUT,
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msopst.main()
        self.assertEqual(
            error.value.code, ConstManager.OP_TEST_GEN_NONE_ERROR)

    def test_create_cmd_for_aicpu_from_ini_file(self):
        """
        test create cmd of aicpu support parse head file to create case.json
        """
        test_utils.clear_out_path(ST_OUTPUT)

        args = ['msopst', 'create', '-i', LESS_INI_INPUT,
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msopst.main()
        self.assertEqual(
            error.value.code, ConstManager.OP_TEST_GEN_NONE_ERROR)

    # ------------------------paramType is optional---------------
    def test_create_cmd_form_with_optional_ini(self):
        """
        test create cmd of paramType is optional
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'create', '-i', OPTIONAL_INI_INPUT,
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopst.main()
        self.assertTrue(test_utils.check_file_context(
            ST_OUTPUT, OPTIONAL_ST_GOLDEN_OUTPUT))

    def test_gen_optional_acl_src_code(self):
        """
        test run cmd of paramType is optional
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_OP_CASE_JSON_INPUT, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()
        pooling_output_testcase = os.path.join(ST_OUTPUT, 'Pooling/src/testcase.cpp')
        pooling_output_json = os.path.join(ST_OUTPUT,
                                           'Pooling/run/out/test_data/config/')
        self.assertTrue(compare_context(
            pooling_output_testcase, ST_GOLDEN_ACL_PROJECT_OUTPUT_TESTCASE))
        self.assertTrue(test_utils.check_file_context(
            pooling_output_json, ST_GOLDEN_ACL_PROJECT_OUTPUT_RUN))

    # ------------------------dynamic_shape--------------------
    def test_gen_dynamic_shape_src_code(self):
        """
        test generate acl code of dynamic shape
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'mi', 'gen_testcase', '-i',
                ST_GOLDEN_OP_ADD_REPORT_JSON_INPUT, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils'
                        '.check_path_valid'):
                    msopst.main()
        src_output_testcase = os.path.join(ST_OUTPUT, 'src/testcase.cpp')
        self.assertTrue(compare_context(
            src_output_testcase, ST_GOLDEN_DYNAMIC_SHAPE_TESTCASE))

    # ---------------------------Mindspore----------------------
    @unittest.mock.patch('op_test_frame.st.interface.case_generator.getattr')
    def test_create_cmd_for_mindspore(self, getattr_mock):
        """
        test create cmd for mindspore operator
        """
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

    def test_gen_mindspore_src_code(self):
        """
        test generate test script for mindspore operator
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'mi', 'gen_testcase', '-i', ST_MS_GOLDEN_INPUT_JSON, '-out',
                ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils'
                        '.check_path_valid'):
                    with mock.patch('op_test_frame.st.interface'
                                    '.ms_op_generator'
                                    '.importlib.import_module'):
                        msopst.main()
        self.assertEqual(error.value.code,
                         ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)

    def test_ms_op_generator(self):
        """
        test _create_ms_op_json_content function of ms_op_generator.py
        """
        data = [{"case_name": "Test_Add_001",
                 "op": "Add",
                 "st_mode": 'ms_python_train',
                 "calc_expect_func_file": "/home/aa.py"
                                          ":calc_expect_func",
                 "input_desc": [
                     {
                         "format": ["NC1HWC0"],
                         "ori_format": ["NCHW"],
                         "type": "float16",
                         "shape": [8, 1, 16, 4, 16],
                         "ori_shape": [8, 1, 16, 4, 16],
                         "data_distribute": ["uniform"],
                         "value_range": [[0.1, 1.0]]
                     }],
                 "output_desc": [
                     {
                         "format": ["NC1HWC0"],
                         "ori_format": ["NCHW"],
                         "type": "float16",
                         "shape": [8, 1, 16, 4, 16],
                         "ori_shape": [8, 16, 16, 4]
                     }]
                 }]
        ms_op_generator._create_ms_op_json_content(data)

    @staticmethod
    def mock_gen_data(cls, *args, **kwargs):
        """mock tofile"""
        return NumpyArrar()

    def test_gen_data_with_value_code(self):
        """
        test tbe gen data with value
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_OP_GEN_WITH_VALUE_JSON_INPUT, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()
        adds_output_src = os.path.join(ST_OUTPUT, 'Adds/src')
        self.assertTrue(test_utils.check_file_context(
            adds_output_src, ST_GOLDEN_OP_GEN_WITH_VALUE_ACL_PROJECT_OUTPUT))

    def test_ms_gen_data_with_value_code(self):
        """
        test mindspore gen data with value
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_MS_GOLDEN_INPUT_JSON_WITH_VALUE, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('op_test_frame.st.interface'
                                '.ms_op_generator'
                                '.importlib.import_module'):
                    msopst.main()
        try:
            self.assertEqual(
                error.value.code, ConstManager.OP_TEST_GEN_NONE_ERROR)
        finally:
            test_utils.clear_out_path(ST_OUTPUT)

    def test_gen_fuzz_acl_src_code(self):
        """
        test run cmd of paramType is optional
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_OP_FUZZ_CASE_JSON_INPUT, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()
        fuzz_add_output_src = os.path.join(ST_OUTPUT, 'Add/src/testcase.cpp')
        fuzz_add_output_json = os.path.join(ST_OUTPUT,
                                           'Add/run/out/test_data/config/')
        self.assertTrue(compare_context(
            fuzz_add_output_src, ST_GOLDEN_OP_FUZZ_CASE_OUTPUT_SRC))
        self.assertTrue(test_utils.check_file_context(
            fuzz_add_output_json, ST_GOLDEN_OP_FUZZ_CASE_OUTPUT_RUN))

    def test_gen_fuzz_ms_src_code(self):
        """
        test mindspore gen data with value
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_MS_FUZZ_CASE_JSON_INPUT, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch('op_test_frame.st.interface'
                                '.ms_op_generator.MsOpGenerator'
                                '._get_mindspore_input_param_type'):
                    msopst.main()
        fuzz_square_output_src = os.path.join(ST_OUTPUT, 'Square/src')
        self.assertTrue(test_utils.check_file_context(
            fuzz_square_output_src, ST_GOLDEN_MS_FUZZ_CASE_OUTPUT_SRC))

    def test_const_input_with_value(self):
        """
        test const input with value, run st
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_CONST_INPUT_VALUE_JSON, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()
        const_testcase_cpp = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/src/testcase.cpp')
        const_op_execute_cpp = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/src/op_execute.cpp')
        const_acl_op_json = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/run/out/test_data'
                                                    '/config/acl_op.json')
        self.assertTrue(compare_context(
            const_testcase_cpp, ST_GOLDEN_CONST_INPUT_SRC_TESTCASE))
        self.assertTrue(compare_context(
            const_op_execute_cpp, ST_GOLDEN_CONST_INPUT_SRC_OP_EXECUTE))
        self.assertTrue(compare_context(
            const_acl_op_json, ST_GOLDEN_CONST_INPUT_CONFIG_ACL_OP))

    def test_const_input_with_data_distribute(self):
        """
        test const input with data_distribute and value_range, run st
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_CONST_INPUT_DATA_DISTRIBUTE_JSON, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()
        const_testcase_cpp = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/src/testcase.cpp')
        const_op_execute_cpp = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/src/op_execute.cpp')
        const_acl_op_json = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/run/out/test_data'
                                                    '/config/acl_op.json')
        self.assertTrue(compare_context(
            const_testcase_cpp, ST_GOLDEN_CONST_INPUT_SRC_TESTCASE))
        self.assertTrue(compare_context(
            const_op_execute_cpp, ST_GOLDEN_CONST_INPUT_SRC_OP_EXECUTE))
        self.assertTrue(compare_context(
            const_acl_op_json, ST_GOLDEN_CONST_INPUT_CONFIG_ACL_OP))

    def test_abnormal_case(self):
        """
        test abnormal case with input json
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_ABNORMAL_CASE_JSON_INPUT, '-soc',
                'Ascend310', '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()

    def test_const_input_with_no_value(self):
        """
        test const input without value, default data_distribute and value_range,
        run st
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_CONST_INPUT_NO_VALUE_JSON, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()
        const_testcase_cpp = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/src/testcase.cpp')
        const_op_execute_cpp = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/src/op_execute.cpp')
        const_acl_op_json = os.path.join(ST_OUTPUT, 'ResizeBilinearV2/run/out/test_data'
                                                    '/config/acl_op.json')
        self.assertTrue(compare_context(
            const_testcase_cpp, ST_GOLDEN_CONST_INPUT_SRC_TESTCASE))
        self.assertTrue(compare_context(
            const_op_execute_cpp, ST_GOLDEN_CONST_INPUT_SRC_OP_EXECUTE))
        self.assertTrue(compare_context(
            const_acl_op_json, ST_GOLDEN_CONST_INPUT_CONFIG_ACL_OP))

    def test_scalar_input_with_value(self):
        """
        run st: scalar input with value
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'run', '-i', ST_GOLDEN_SCALAR_INPUT_WITH_VALUE_JSON, '-soc',
                'Ascend310', '-conf', MSOPST_CONF_INI, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch('op_test_frame.st.interface.utils.execute_command'):
                    msopst.main()
        const_testcase_cpp = os.path.join(ST_OUTPUT, 'TestScalar/src/testcase.cpp')
        const_acl_op_json = os.path.join(ST_OUTPUT, 'TestScalar/run/out/test_data'
                                                    '/config/acl_op.json')
        self.assertTrue(compare_context(
            const_testcase_cpp, ST_GOLDEN_SCALAR_INPUT_SRC_TESTCASE))
        self.assertTrue(compare_context(
            const_acl_op_json, ST_GOLDEN_SCALAR_INPUT_CONFIG_ACL_OP))

    def test_check_list_float(self):
        err_thr = utils.check_list_float([0.1, 0.1], "err_thr")
        self.assertEqual(err_thr,[0.1, 0.1])

    def test_check_list_float_error(self):
        with pytest.raises(utils.OpTestGenException) as error:
            utils.check_list_float(["A"], "err_thr")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_ERROR_THRESHOLD_ERROR)

    def test_arg_parser(self):
        argument = ['msopst', 'create', '-i', CAST_INI_INPUT,
                '-out', ST_OUTPUT]
        with mock.patch('sys.argv', argument):
            args = MsopstArgParser()
            args.get_input_file()
            args.get_output_path()

    def test_clas_report_json_encoder(self):
        value1 = np.int8(1)
        value2 = np.float16(1.0)
        value4 = np.zeros(5)
        json_obj ={"value1": [value1],
                   "value2": [value2],
                   "value3": [(1.000000066 + 0j)],
                   "value4": [value4]
                   }
        json.dumps(json_obj, indent=4, cls=ReportJsonEncoder)

    def test_arg_parser_expection(self):
        argument1 = ['msopst']
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('sys.argv', argument1):
                args = MsopstArgParser()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
        argument2 = ['msopst', 'mi']
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('sys.argv', argument2):
                args = MsopstArgParser()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
        argument3 = ['msopst', 'mi', 'gen_testcase', '-i', ST_GOLDEN_OP_POOLING_REPORT_JSON_INPUT,
                     '-d', 'test']
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('sys.argv', argument3):
                args = MsopstArgParser()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_DEVICE_ID_ERROR)

    def test_msopst_write_content_to_file_error(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.fdopen', side_effect=OSError):
                AtcTransformOm._write_content_to_file("content", "/home")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)

    def test_msopst_write_json_file1(self):
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('os.fdopen', side_effect=IOError):
                utils.write_json_file('/home/result', "test")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)


if __name__ == '__main__':
    unittest.main()
