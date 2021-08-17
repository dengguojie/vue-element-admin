import sys
import unittest
import pytest
from unittest import mock
import os
from op_gen.interface import utils
from op_gen import msopgen
sys.path.append(os.path.dirname(__file__)+"/../../")
from util import test_utils


OUT_PATH_VALID = './st/msopgen/res/tmp'
IR_JSON_PATH = './st/msopgen/res/IR_json.json'
MS_JSON_PATH = './st/msopgen/res/MS_json.json'
IR_EXCEL_PATH = './st/msopgen/res/IR_excel.xlsx'

IR_JSON_GOLDEN_OUTPUT = './st/msopgen/golden/golden_from_json/golden_output'
IR_JSON_OUTPUT = './st/msopgen/golden/golden_from_json/output'

MS_JSON_GOLDEN_OUTPUT = './st/msopgen/golden/golden_from_ms_json/golden_output'
MS_JSON_OUTPUT = './st/msopgen/golden/golden_from_ms_json/output'

MS_SQUARE_INPUT = './st/msopgen/golden/golden_from_ms_txt/input/square.txt'
MS_SUM_INPUT = './st/msopgen/golden/golden_from_ms_txt/input/sum.txt'

MS_TXT_GOLDEN_OUTPUT = './st/msopgen/golden/golden_from_ms_txt/golden_output'
MS_TXT_OUTPUT = './st/msopgen/golden/golden_from_ms_txt/output'

IR_EXCEL_GOLDEN_OUTPUT = './st/msopgen/golden/golden_from_excel/golden_output'
IR_EXCEL_OUTPUT = './st/msopgen/golden/golden_from_excel/output'


class TestUtilsMethods(unittest.TestCase):
    def test_mi_query_from_ir_json_success(self):
        test_utils.clear_out_path(OUT_PATH_VALID)
        args = ['msopgen.py', 'mi', 'query', '-i', IR_JSON_PATH,
                '-out', OUT_PATH_VALID]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msopgen.main()
        self.assertEqual(error.value.code, utils.MS_OP_GEN_NONE_ERROR)

    def test_mi_query_from_ir_excel_success(self):
        test_utils.clear_out_path(OUT_PATH_VALID)
        args = ['msopgen.py', 'mi', 'query', '-i', IR_EXCEL_PATH,
                '-out', OUT_PATH_VALID]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msopgen.main()
        self.assertEqual(error.value.code, utils.MS_OP_GEN_NONE_ERROR)

    def test_gen_tf_caffe_onnx_from_ir_json_compare_success(self):
        test_utils.clear_out_path(IR_JSON_OUTPUT)
        args = ['msopgen.py', 'gen', '-i', IR_JSON_PATH, '-f', 'tf', '-c',
                'ai_core-ascend310', '-op', 'Conv2D', '-out', IR_JSON_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopgen.main()
        args1 = ['msopgen.py', 'gen', '-i', IR_JSON_PATH, '-f', 'caffe', '-c',
                 'ai_core-ascend310', '-op', 'Conv2D', '-out', IR_JSON_OUTPUT,
                 '-m', '1']
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args1):
                msopgen.main()
        args2 = ['msopgen.py', 'gen', '-i', IR_JSON_PATH, '-f', 'caffe', '-c',
                 'aicpu', '-op', 'Conv2D', '-out', IR_JSON_OUTPUT,
                 '-m', '1']
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args2):
                msopgen.main()
        args3 = ['msopgen.py', 'gen', '-i', IR_JSON_PATH, '-f', 'onnx', '-c',
                 'aicpu', '-op', 'Conv2D', '-out', IR_JSON_OUTPUT,
                 '-m', '1']
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args3):
                msopgen.main()
        args4 = ['msopgen.py', 'gen', '-i', IR_JSON_PATH, '-f', 'onnx', '-c',
                 'ai_core-ascend310', '-op', 'Conv2D', '-out', IR_JSON_OUTPUT,
                 '-m', '1']
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args4):
                msopgen.main()
        self.assertTrue(test_utils.check_result(IR_JSON_OUTPUT,
                                                IR_JSON_GOLDEN_OUTPUT))

    def test_gen_ms_aicore_tf_from_ms_json_compare_success(self):
        test_utils.clear_out_path(MS_JSON_OUTPUT)
        args = ['msopgen.py', 'gen', '-i', MS_JSON_PATH, '-f', 'MS', '-c',
                'ai_core-ascend310', '-op', 'Conv2D', '-out', MS_JSON_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopgen.main()
        self.assertTrue(test_utils.check_result(MS_JSON_OUTPUT,
                                                MS_JSON_GOLDEN_OUTPUT))

    def test_gen_ms_aicore_from_tf_txt_compare_success(self):
        test_utils.clear_out_path(MS_TXT_OUTPUT)
        args = ['msopgen.py', 'gen', '-i', MS_SQUARE_INPUT, '-f', 'ms', '-c',
                'ai_core-ascend310', '-out', MS_TXT_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopgen.main()
        args1 = ['msopgen.py', 'gen', '-i', MS_SUM_INPUT, '-f', 'mindspore',
                 '-c', 'ai_core-ascend310', '-out', MS_TXT_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args1):
                msopgen.main()
        self.assertTrue(test_utils.check_result(MS_TXT_OUTPUT,
                                                MS_TXT_GOLDEN_OUTPUT))

    def test_gen_tf_from_ir_excel_compare_success(self):
        test_utils.clear_out_path(IR_EXCEL_OUTPUT)
        args = ['msopgen.py', 'gen', '-i', IR_EXCEL_PATH, '-f', 'tf', '-c',
                'ai_core-ascend310', '-op', 'Conv2DTik', '-out', IR_EXCEL_OUTPUT]
        with pytest.raises(SystemExit):
            with mock.patch('sys.argv', args):
                msopgen.main()
        self.assertTrue(test_utils.check_result(IR_EXCEL_OUTPUT,
                                                IR_EXCEL_GOLDEN_OUTPUT))

if __name__ == '__main__':
    unittest.main()