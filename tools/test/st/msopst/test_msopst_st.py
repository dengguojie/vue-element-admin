import json
import sys
import unittest

import numpy as np
import pytest
from unittest import mock
import os
import importlib.util
import importlib.machinery
from op_test_frame.st.interface import utils
from op_test_frame.st.interface import model_parser
from op_test_frame.st.interface.framework import tf_model_parser
from op_test_frame.st.interface.framework.tf_model_parser import TFModelParse
from op_test_frame.st.interface import result_comparer
from op_test_frame.st.interface.case_generator import CaseGenerator
from op_test_frame.st.interface.data_generator import DataGenerator
from op_test_frame.st.interface.st_report import OpSTReport
from op_test_frame.st.interface.st_report import OpSTCaseReport
from op_test_frame.st.interface.op_st_case_info import OpSTCase
from op_test_frame.st.interface.op_st_case_info import OpSTCaseTrace
from op_test_frame.st.interface.acl_op_runner import AclOpRunner
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
ST_GOLDEN_ACL_PROJECT_OUTPUT_SRC = './st/msopst/golden/base_case/golden_output' \
                                   '/gen_optional_acl_prj/Pooling/src'
ST_GOLDEN_ACL_PROJECT_OUTPUT_RUN = './st/msopst/golden/base_case/golden_output' \
                                   '/gen_optional_acl_prj/Pooling/run/out'\
                                   '/test_data/config/'
MSOPST_CONF_INI = './st/msopst/golden/base_case/input/msopst.ini'

# dynamic shape
ST_GOLDEN_OP_DYNAMIC_SHAPE_INI_INPUT = './st/msopst/golden/base_case/input' \
                                       '/add.ini'

ST_GOLDEN_OP_DYNAMIC_SHAPE_JSON_INPUT = './st/msopst/golden/base_case/input' \
                                        '/add_case.json'
ST_GOLDEN_DYNAMIC_SHAPE_ACL_PROJECT_OUTPUT = './st/msopst/golden/base_case' \
                                             '/golden_output/' \
                                             'gen_optional_acl_prj/Add/src'

# MINDSPORE OPERATOR INPUT/OUTPOUT
ST_MS_GOLDEN_JSON_OUTPUT = './st/msopst/golden/base_case/golden_output' \
                           '/mindspore/json'
ST_MS_GOLDEN_INPUT_JSON = './st/msopst/golden/base_case/input/ms_case.json'


class NumpyArrar:
    def tofile(self, file_path):
        pass


class Args:
    def __init__(self, input_file, output_path, model_path):
        self.input_file = input_file
        self.output_path = output_path
        self.model_path = model_path
        self.quiet = False


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
                         utils.OP_TEST_GEN_INVALID_PATH_ERROR)

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
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)

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
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)

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
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)

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
                         utils.OP_TEST_GEN_INVALID_PATH_ERROR)

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
                         utils.OP_TEST_GEN_TF_CHANGE_PLACEHOLDER_ERROR)

    def test_model_parser_4(self):
        """
        verify the abnormal scene of get_model_nodes function
        in model_parser.py
        """
        args = Args('./a.json', '', './a.pbv')
        with pytest.raises(utils.OpTestGenException) as error:
            model_parser.get_model_nodes(args, '')
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_INVALID_PARAM_ERROR)

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
                         utils.OP_TEST_GEN_TF_LOAD_ERROR)

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
                            'op_test_frame.st.interface.acl_op_generator'
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
                                    with mock.patch('os.makedirs'):
                                        with mock.patch('os.chmod'):
                                            open_file.write = None
                                            msopst.main()
        self.assertEqual(
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)
        DataGenerator.gen_data = self.gen_data_

    def test_compare_func_1(self):
        """
        verify the abnormal scene of compare function in result_comparer.py
        """
        report = OpSTReport()
        run_dir = "xxx.txt"
        result_comparer.compare(report, run_dir)

    def test_compare_func_2(self):
        """
        verify the normal scene of compare function in result_comparer.py
        """
        report = OpSTReport()
        op_st = OpSTCase("AddN", {"calc_expect_func_file_func": 1})
        op_st_case_trace = OpSTCaseTrace(op_st)
        op_st_report = OpSTCaseReport(op_st_case_trace)
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner'
                    '.AclOpRunner._execute_command'):
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
                                runner.process()
                                result_comparer.compare(
                                    report, ST_GOLDEN_OP_RESULT_TXT)

    def test_compare_func_3(self):
        """
        verify the normal scene of compare2 function in result_comparer.py
        """
        report = OpSTReport()
        op_st = OpSTCase("AddN", {"calc_expect_func_file_func": 1})
        op_st_case_trace = OpSTCaseTrace(op_st)
        op_st_report = OpSTCaseReport(op_st_case_trace)
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch(
                    'op_test_frame.st.interface.acl_op_runner'
                    '.AclOpRunner._execute_command'):
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
                                    runner.process()
                                    result_comparer.compare2(
                                        report, ST_GOLDEN_OP_RESULT_TXT)

    def test_compare_func_4(self):
        """
        verify the normal scene of data_compare function in result_comparer.py
        """
        npu_output = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        cpu_output = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
        npu_output_array = np.array(npu_output)
        cpu_output_array = np.array(cpu_output)
        result_comparer._data_compare(npu_output_array, cpu_output_array)

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
                             utils.OP_TEST_GEN_WRITE_FILE_ERROR)

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
        verify the normal scene of _prof_get_op_time_from_csv_file function
        in acl_op_runner.py
        """
        csv_file = "./st/msopst/golden/base_case/input/task_time_0_1.csv"
        op_name_list = ["Less", "Less"]
        soc_version = "Ascend310"
        report = OpSTReport()
        runner = AclOpRunner('/home', 'ddd', report)
        runner._prof_get_op_time_from_csv_file(
            csv_file, op_name_list, soc_version)

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

    # --------------ori_format/ori_shape/device_id--------------------
    def test_gen_ori_format_or_shape_src_code(self):
        """
        test generate acl code for ori_format/ori_shape/device_id
        of TBE operators
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'mi', 'gen', '-i', ST_GOLDEN_OP_CASE_JSON_INPUT,
                '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils'
                        '.check_path_valid'):
                    msopst.main()
        op_output_src = os.path.join(ST_OUTPUT, 'src')
        op_output_run = os.path.join(ST_OUTPUT, 'run/out/test_data/config/')
        self.assertTrue(test_utils.check_file_context(
            op_output_src, ST_GOLDEN_ACL_PROJECT_OUTPUT_SRC))
        self.assertTrue(test_utils.check_file_context(
            op_output_run, ST_GOLDEN_ACL_PROJECT_OUTPUT_RUN))

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
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)

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
            error.value.code, utils.OP_TEST_GEN_NONE_ERROR)

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
                        'op_test_frame.st.interface.utils'
                        '.check_path_valid'):
                    msopst.main()
        pooling_output_src = os.path.join(ST_OUTPUT, 'Pooling/src')
        pooling_output_json = os.path.join(ST_OUTPUT,
                                           'Pooling/run/out/test_data/config/')
        self.assertTrue(test_utils.check_file_context(
            pooling_output_src, ST_GOLDEN_ACL_PROJECT_OUTPUT_SRC))
        self.assertTrue(test_utils.check_file_context(
            pooling_output_json, ST_GOLDEN_ACL_PROJECT_OUTPUT_RUN))

    # ------------------------dynamic_shape--------------------
    def test_gen_dynamic_shape_src_code(self):
        """
        test generate acl code of dynamic shape
        """
        test_utils.clear_out_path(ST_OUTPUT)
        args = ['msopst', 'mi', 'gen', '-i',
                ST_GOLDEN_OP_DYNAMIC_SHAPE_JSON_INPUT, '-out', ST_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                with mock.patch(
                        'op_test_frame.st.interface.utils'
                        '.check_path_valid'):
                    msopst.main()
        src_output = os.path.join(ST_OUTPUT, 'src')
        self.assertTrue(test_utils.check_file_context(
            src_output, ST_GOLDEN_DYNAMIC_SHAPE_ACL_PROJECT_OUTPUT))

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
        args = ['msopst', 'mi', 'gen', '-i', ST_MS_GOLDEN_INPUT_JSON, '-out',
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
                         utils.OP_TEST_GEN_WRITE_FILE_ERROR)

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


if __name__ == '__main__':
    unittest.main()
