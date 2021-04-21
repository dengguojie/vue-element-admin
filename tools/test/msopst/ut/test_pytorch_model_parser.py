import io
import sys
import unittest
from unittest.mock import patch
from unittest import mock
from op_test_frame.st.interface.framework import pytorch_model_parser


class Model:
    pass


class Graph:
    def __init__(self):
        self.input = 1
        self.value_info = 1


class Tensor:
    def __init__(self):
        self.dims = 1
        self.data_type = 1


class Type:
    pass


class Tensor_Type:
    pass


class Shape:
    pass


class Dim:
    def __init__(self):
        self.dim_param = "1"
        self.dim_value = 1


class Node:
    def __init__(self):
        self.name = "1"
        self.input = 1
        self.output = 2


class Attr:
    def __init__(self):
        self.name = "1"
        self.type = 2


class Args:
    def __init__(self):
        self.input_file = "a.json"
        self.output_path = ""
        self.model_path = "./"
        self.quiet = ""


class test_pytorch_model_parser(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_check_shape_and_notice(self):
        shape = [1, 2, 3, 4]
        out = pytorch_model_parser.check_shape_and_notice(shape)
        self.assertTrue(out is False)
        shape = "1w,2,3,4"
        out = pytorch_model_parser.check_shape_and_notice(shape)
        self.assertTrue(out is False)
        shape = "0,2,3,4"
        out = pytorch_model_parser.check_shape_and_notice(shape)
        self.assertTrue(out is False)
        shape = "1,2,3,4"
        out = pytorch_model_parser.check_shape_and_notice(shape)
        self.assertTrue(out is True)

    def test__get_tensor_shape(self):
        tensor_names = ["a", "b"]
        all_tensors = {"a": {'dtype': 1, 'shape': [1, 3, 224, 224]}}
        dtypes, shapes = pytorch_model_parser._get_tensor_shape(tensor_names, all_tensors)
        print("\n\n\n\n\n")
        print("znnnnnnnnnnnnnnnnn")
        print("dtypes:", dtypes)
        print("shapes:", shapes)
        print("\n\n\n\n\n")
        self.assertTrue(dtypes == [1])
        self.assertTrue(shapes == [[1, 3, 224, 224]])

    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           "._update_tensor_info")
    def test__parse_tensor_info(self, mock__update_tensor_info):
        mock__update_tensor_info.return_value = "1"
        all_tensors = {"b": 1}
        tensor1 = Tensor()
        tensor2 = Tensor()
        tensor3 = Tensor()
        type1 = Type()
        type2 = Type()
        tensor_type1 = Tensor_Type()
        tensor_type1.elem_type = "int32"
        tensor_type2 = Tensor_Type()
        tensor_type2.elem_type = "int32"
        tensor_type2.shape = ""
        tensor1.name = "a"
        tensor2.name = "b"
        tensor3.name = "c"
        type1.tensor_type = tensor_type1
        type2.tensor_type = tensor_type2
        tensor1.type = type1
        tensor3.type = type2
        out = pytorch_model_parser._parse_tensor_info(
            [tensor1, tensor2, tensor3], all_tensors)
        self.assertTrue(out == {'b': 1})

        shape = Shape()
        dim1 = Dim()
        dim2 = Dim
        shape.dim = [dim1, dim2]
        tensor_type2.shape = shape
        type2.tensor_type = tensor_type2
        tensor3.type = type2
        out = pytorch_model_parser._parse_tensor_info(
            [tensor1, tensor2, tensor3], all_tensors)
        self.assertTrue(out == {'b': 1})

    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           "._parse_tensor_info")
    def test__get_all_tensors(self, mock__parse_tensor_info):
        tensor1 = Tensor()
        tensor2 = Tensor()
        graph = Graph()
        tensor1.name = "a"
        tensor2.name = "b"
        graph.initializer = [tensor1, tensor2]
        mock__parse_tensor_info.return_value = {"a": 1}
        out = pytorch_model_parser._get_all_tensors(graph)
        self.assertTrue(out == {'a': 1, 'b': {'dtype': 'float', 'shape': 1}})

    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           ".check_shape_and_notice")
    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           ".get_shape_and_notice")
    def test__get_input_shape_from_user(self, mock_get_shape_and_notice,
                                        mock_check_shape_and_notice):
        new_shape_map = {"a": {"ori_shape": [1, 2, 3, 4]}}
        mock_get_shape_and_notice.return_value = None
        sys.stdin = io.StringIO("n\n")
        out = pytorch_model_parser._get_input_shape_from_user(new_shape_map)
        self.assertTrue(out is False)

        mock_check_shape_and_notice.return_value = True
        sys.stdin = io.StringIO("m,m\n")
        out = pytorch_model_parser._get_input_shape_from_user(new_shape_map)
        self.assertTrue(out is True)

    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           "._get_tensor_shape")
    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           "._get_all_tensors")
    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           "._infer_model_shape")
    @patch("op_test_frame.st.interface.framework.pytorch_model_parser._load_model")
    @patch("onnx.helper.get_attribute_value")
    def test__get_node_list(self, mock_get_attribute_value, mock__load_model,
                            mock__infer_model_shape, mock__get_all_tensors,
                            mock__get_tensor_shape):
        model_path = ""
        ini_op_type = "1"
        mock__load_model.return_value = ""
        model = Model()
        graph = Graph()
        node1 = Node()
        node2 = Node()
        attr1 = Attr()
        attr2 = Attr()
        node1.op_type = "1"
        node1.attribute = [attr1, attr2]
        node2.op_type = "2"
        graph.node = [node1, node2]
        model.graph = graph
        mock__infer_model_shape.return_value = model
        mock__get_all_tensors.return_value = []
        mock__get_tensor_shape.return_value = ("int32", [])
        mock_get_attribute_value.return_value = ""
        out = pytorch_model_parser._get_node_list(model_path, ini_op_type)
        self.assertTrue(out is not None)

    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           "._parse_tensor_info")
    @patch("op_test_frame.st.interface.framework.pytorch_model_parser._load_model")
    def test__get_model_inputs(self, mock__load_model,
                               mock__parse_tensor_info):
        model = Model()
        graph = Graph()
        tensor = Tensor()
        tensor.name = "a"
        graph.initializer = [tensor]
        model.graph = graph
        mock__load_model.return_value = model
        mock__parse_tensor_info.return_value = {"a": {"shape": ""},
                                                "b": {"shape": ""}}
        args = Args()
        pytorch_model = pytorch_model_parser.PyTorchModelParse(args)
        out = pytorch_model._get_model_inputs()
        self.assertTrue(out == {'b': {'ori_shape': '', 'new_shape': []}})

    @patch("onnx.save")
    @patch("onnx.shape_inference.infer_shapes")
    @patch("onnx.helper.make_tensor_value_info")
    @patch("onnx.checker.check_model")
    @patch("op_test_frame.st.interface.framework.pytorch_model_parser._load_model")
    def test__change_shape_fn(self, mock__load_model, mock_check_model,
                              mock_make_tensor_value_info, mock_infer_shapes,
                              mock_save):
        model = Model()
        graph = Graph()
        tensor1 = Tensor()
        tensor2 = Tensor()
        type1 = Type()
        tensor_type = Tensor_Type()
        tensor_type.elem_type = "int8"
        type1.tensor_type = tensor_type
        tensor1.type = type1
        tensor1.name = "a"
        tensor2.name = "b"
        graph.input = [tensor1, tensor2]
        model.graph = graph
        mock__load_model.return_value = model
        mock_check_model.return_value = ""
        mock_make_tensor_value_info.return_value = ""
        mock_infer_shapes.return_value = ""
        mock_save.return_value = ""

        new_shape_map = {"a": {"new_shape": [1, 2, 3]}}
        args = Args()
        pytorch_model = pytorch_model_parser.PyTorchModelParse(args)
        out = pytorch_model._change_shape_fn(new_shape_map)
        self.assertTrue(out is not None)

    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           "._get_node_list")
    @patch("op_test_frame.st.interface.framework.pytorch_model_parser"
           "._get_input_shape_from_user")
    @patch("op_test_frame.st.interface.utils.check_name_valid")
    @patch("op_test_frame.st.interface.utils.check_path_valid")
    def test_get_model_nodes(self, mock_check_path_valid,
                             mock_check_name_valid,
                             mock__get_input_shape_from_user,
                             mock__get_node_list):
        mock_check_path_valid.return_value = ""
        mock_check_name_valid.return_value = ""
        mock__get_input_shape_from_user.return_value = True
        mock__get_node_list.return_value = []
        ini_op_type = "int8"
        args = Args()
        pytorch_model = pytorch_model_parser.PyTorchModelParse(args)
        pytorch_model._get_model_inputs = mock.Mock(return_value='')
        pytorch_model._change_shape_fn = mock.Mock(return_value='')
        out = pytorch_model.get_model_nodes(ini_op_type)
        self.assertTrue(out == [])
        mock__get_input_shape_from_user.return_value = False
        out = pytorch_model.get_model_nodes(ini_op_type)
        self.assertTrue(out == [])
        pytorch_model.quiet_flag = True
        out = pytorch_model.get_model_nodes(ini_op_type)
        self.assertTrue(out == [])


if __name__ == '__main__':
    unittest.main()
