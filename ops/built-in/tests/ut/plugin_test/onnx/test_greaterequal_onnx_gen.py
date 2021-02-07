import onnx
from onnx import helper
from onnx import TensorProto


def make_greater_equal():
    """same shape"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2, 6])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 2, 6])
    node_def = helper.make_node(
        'GreaterOrEqual',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    graph = helper.make_graph(
        [node_def],
        'GreaterOrEqual',
        [x, y],
        [z],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_greater_equal_case_1.onnx")
    onnx.checker.check_model(model)


def make_greater_equal1():
    """broadcast"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 1, 6])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 3, 6])
    node_def = helper.make_node(
        'GreaterOrEqual',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    graph = helper.make_graph(
        [node_def],
        'GreaterOrEqual',
        [x, y],
        [z],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_greater_equal_case_2.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_greater_equal()
    make_greater_equal1()
