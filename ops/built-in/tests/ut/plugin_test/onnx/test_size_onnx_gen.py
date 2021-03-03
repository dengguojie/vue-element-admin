import onnx
from onnx import helper
from onnx import TensorProto


def make_size_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1, ])
    node_def = helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Size_v11',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_size_case_v11.onnx")
    onnx.checker.check_model(model)


def make_size_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1, ])
    node_def = helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Size_v12',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_size_case_v12.onnx")
    onnx.checker.check_model(model)


def make_size_v13():
    """v13"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1, ])
    node_def = helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Size_v13',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_size_case_v13.onnx")
    onnx.checker.check_model(model)


def make_size_v9():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [1, ])
    node_def = helper.make_node(
        'Size',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Size_v9',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_size_case_v9.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_size_v9()
    make_size_v11()
    make_size_v12()
    make_size_v13()
