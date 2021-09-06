import unittest
import pytest
from unittest import mock
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager
from op_test_frame.st.interface.framework.tf_model_parser import TFModelParse
from op_test_frame.st.interface.framework import tf_model_parser


INPUT_PATH_VALID = './st/msopst/res/tmp/conv2_d.ini'
INPUT_PATH_INVALID = './st/msopst/res/tmp/a.ini'
INPUT_JSON_PATH_VALID = './st/msopst/res/tmp/tmp_shape.json'
INPUT_JSON_PATH_INVALID = './st/msopst/res/tmp/a.json'
OUTPUT_PATH_VALID = './st/msopst/res/tmp/'
OUTPUT_PATH_INVALID = './msa-2!@#$%t/res/tmp/'
MODEL_PATH_VALID = './st/msopst/res/tmp/resnet50.pb'
MODEL_PATH_INVALID = './st/msopst/res/tmp/b.pb'

class Args:
    def __init__(self, input_file, output_path, model_path, quiet_flag):
        self.input_file = input_file
        self.output_path = output_path
        self.model_path = model_path
        self.quiet = quiet_flag


class TestUtilsMethods(unittest.TestCase):
    def test_msopst_get_tf_model_node_success01(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, True)
        tf_model_parser = TFModelParse(args)
        tf_model_parser.get_tf_model_nodes('Conv2D')

    def test_msopst_get_tf_model_node_success02(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, True)
        tf_model_parser.get_model_nodes(args, 'Conv2D')

    def test_msopst_get_tf_model_node_error01(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_INVALID, True)
        with pytest.raises(utils.OpTestGenException) as error:
            tf_model_parser = TFModelParse(args)
            tf_model_parser.get_tf_model_nodes('Conv2D')
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_map_tf_input_output_dtype(self):
        tf_dtype_list = ["float32", "float16", "int8", "int16", "int32",
                         "uint8", "uint16", "uint32", "bool", "int64"]
        for tf_dtype in tf_dtype_list:
            tf_model_parser._map_tf_input_output_dtype(tf_dtype)

    def test_msopst_get_shape_success01(self):
        args = Args("", OUTPUT_PATH_VALID, MODEL_PATH_VALID, "")
        tf_model_parser = TFModelParse(args)
        tf_model_parser.get_shape()

    def test_msopst_get_shape_success02(self):
        args = Args("", OUTPUT_PATH_VALID, MODEL_PATH_VALID, "")
        tf_model_parser.get_shape(args, "conv2d")

    def test_msopst_get_shape_error01_invalid_model_path(self):
        args = Args("", OUTPUT_PATH_VALID,
                    MODEL_PATH_INVALID, "")
        with pytest.raises(utils.OpTestGenException) as error:
            tf_model_parser = TFModelParse(args)
            tf_model_parser.get_shape()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_get_shape_error02_tf_load_failed(self):
        args = Args("", OUTPUT_PATH_VALID,
                    MODEL_PATH_INVALID, "")
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
                tf_model_parser = TFModelParse(args)
                tf_model_parser.get_shape()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_TF_LOAD_ERROR)

    def test_msopst_get_shape_error03_invalid_output_path(self):
        args = Args("", OUTPUT_PATH_INVALID,
                    MODEL_PATH_INVALID, "")
        with pytest.raises(utils.OpTestGenException) as error:
            tf_model_parser = TFModelParse(args)
            tf_model_parser.get_shape()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_change_shape_success01(self):
        args = Args(INPUT_JSON_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, "")
        lines = '{"Placeholder": {"ori_shape": [-1, 224, 224, 3], "new_shape": [' \
                '224, ' \
                '224,224,3]}}'
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch('builtins.open', mock.mock_open(read_data=lines)):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    open_file.write = None
                    tf_model_parser = TFModelParse(args)
                    tf_model_parser.change_shape()

    def test_msopst_change_shape_success02(self):
        args = Args(INPUT_JSON_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, "")
        lines = '{"Placeholder": {"ori_shape": [-1, 224, 224, 3], "new_shape": [' \
                '224, ' \
                '224,224,3]}}'
        with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
            with mock.patch('builtins.open', mock.mock_open(read_data=lines)):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    open_file.write = None
                    tf_model_parser.change_shape(args, "conv2d")

    def test_msopst_change_shape_no_placeholder01(self):
        args = Args(INPUT_JSON_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, "")
        lines = '{"Placeholder": {"ori_shape": [-1, 224, 224, 3], "new_shape": []}}'
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
                with mock.patch('builtins.open', mock.mock_open(read_data=lines)):
                    with mock.patch('os.open') as open_file, \
                            mock.patch('os.fdopen'):
                        open_file.write = None
                        tf_model_parser = TFModelParse(args)
                        tf_model_parser.change_shape()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_TF_CHANGE_PLACEHOLDER_ERROR)

    def test_msopst_change_shape_no_placeholder02(self):
        args = Args(INPUT_JSON_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, "")
        lines = '{"Unkown": {"ori_shape": [-1, 224, 224, 3], "new_shape": [8,224,224,3]}}'
        with pytest.raises(utils.OpTestGenException) as error:
            with mock.patch('op_test_frame.st.interface.utils.check_path_valid'):
                with mock.patch('builtins.open', mock.mock_open(read_data=lines)):
                    with mock.patch('os.open') as open_file, \
                            mock.patch('os.fdopen'):
                        open_file.write = None
                        tf_model_parser = TFModelParse(args)
                        tf_model_parser.change_shape()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_TF_CHANGE_PLACEHOLDER_ERROR)

    def test_msopst_change_shape_error01_invalid_input_path(self):
        args = Args("./a.txt", OUTPUT_PATH_VALID,
                    MODEL_PATH_INVALID, "")
        with pytest.raises(utils.OpTestGenException) as error:
            tf_model_parser = TFModelParse(args)
            tf_model_parser.change_shape()
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)

    def test_msopst_attr_value_shape_list_error01(self):
        self.assertEqual(tf_model_parser._attr_value_shape_list(None), [])

    def test_msopst_tensor_shape_list_error01(self):
        self.assertEqual(tf_model_parser._tensor_shape_list(None), [])

    def test_msopst_tensor_shape_list_error02(self):
        tensor_shape = [1,2,3]
        self.assertEqual(tf_model_parser._tensor_shape_list(tensor_shape), [[],[],[]])

    def test_msopst_tf_utils_get_operators_error01(self):
        with pytest.raises(utils.OpTestGenException) as error:
            tf_model_parser._tf_utils_get_operators("")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_TF_GET_OPERATORS_ERROR)

    def test_msopst_tf_utils_write_graph_error01(self):
        with pytest.raises(utils.OpTestGenException) as error:
            tf_model_parser._tf_utils_write_graph("", "", "")
        self.assertEqual(error.value.args[0],
                         ConstManager.OP_TEST_GEN_WRITE_FILE_ERROR)

    def test_msopst_check_ori_shape_and_notice_01(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, True)
        tf_model_parser = TFModelParse(args)
        tf_model_parser._check_ori_shape_and_notice([-1,224,224,3], "conv2D")

    def test_msopst_check_ori_shape_and_notice_02(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, True)
        tf_model_parser = TFModelParse(args)
        tf_model_parser._check_ori_shape_and_notice(["a",224,224,3], "conv2D")

    def test_msopst_check_ori_shape_and_notice_03(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, True)
        tf_model_parser = TFModelParse(args)
        tf_model_parser._check_ori_shape_and_notice("a", "conv2D")

    def test_msopst_check_new_shape_and_notice_01(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, True)
        tf_model_parser = TFModelParse(args)
        tf_model_parser._check_new_shape_and_notice("1, a")

    def test_msopst_check_new_shape_and_notice_02(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, True)
        tf_model_parser = TFModelParse(args)
        tf_model_parser._check_new_shape_and_notice("-10, -1")

    def test_msopst_check_new_shape_and_notice_03(self):
        args = Args(INPUT_PATH_VALID, OUTPUT_PATH_VALID,
                    MODEL_PATH_VALID, True)
        tf_model_parser = TFModelParse(args)
        tf_model_parser._check_new_shape_and_notice(1)


if __name__ == '__main__':
    unittest.main()
