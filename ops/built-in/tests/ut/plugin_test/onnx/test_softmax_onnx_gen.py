import onnx
from onnx import helper
from onnx import TensorProto


def make_softmax_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3,4,5,6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])
    node_def = helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
        axis=2
    )

    graph = helper.make_graph(
        [node_def],
        'Softmax_v11',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_softmax_case_v11.onnx")
    onnx.checker.check_model(model)


def make_softmax_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])
    node_def = helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
        axis=1
    )

    graph = helper.make_graph(
        [node_def],
        'Softmax_v12',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_softmax_case_v12.onnx")
    onnx.checker.check_model(model)


def make_softmax_v9():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])
    node_def = helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
        axis=1
    )

    graph = helper.make_graph(
        [node_def],
        'Softmax_v9',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_softmax_case_v9.onnx")
    onnx.checker.check_model(model)


def make_softmax_v13():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])
    node_def = helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Softmax_v13',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_softmax_case_v13.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_softmax_v9()
    make_softmax_v11()
    make_softmax_v12()
    make_softmax_v13()