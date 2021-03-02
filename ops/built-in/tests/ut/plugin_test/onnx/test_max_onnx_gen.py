import onnx
from onnx import helper
from onnx import TensorProto


def make_max_v11():
    """v11"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [3, ])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [3, ])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [3, ])
    d = helper.make_tensor_value_info('d', TensorProto.FLOAT, [3, ])
    node_def = helper.make_node(
        'Max',
        inputs=['a', 'b', 'c'],
        outputs=['d'],
    )

    graph = helper.make_graph(
        [node_def],
        'Max_v11',
        [a, b, c],
        [d],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_max_case_v11.onnx")
    onnx.checker.check_model(model)


def make_max_v12():
    """v12"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [3, ])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [3, ])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [3, ])
    d = helper.make_tensor_value_info('d', TensorProto.FLOAT, [3, ])
    node_def = helper.make_node(
        'Max',
        inputs=['a', 'b', 'c'],
        outputs=['d'],
    )

    graph = helper.make_graph(
        [node_def],
        'Max_v12',
        [a, b, c],
        [d],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_max_case_v12.onnx")
    onnx.checker.check_model(model)


def make_max_v13():
    """v13"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [3, ])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [3, ])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [3, ])
    d = helper.make_tensor_value_info('d', TensorProto.FLOAT, [3, ])
    node_def = helper.make_node(
        'Max',
        inputs=['a', 'b', 'c'],
        outputs=['d'],
    )

    graph = helper.make_graph(
        [node_def],
        'Max_v13',
        [a, b, c],
        [d],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_max_case_v13.onnx")
    onnx.checker.check_model(model)


def make_max_v9():
    """v9"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [3, ])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [3, ])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [3, ])
    d = helper.make_tensor_value_info('d', TensorProto.FLOAT, [3, ])
    node_def = helper.make_node(
        'Max',
        inputs=['a', 'b', 'c'],
        outputs=['d'],
    )

    graph = helper.make_graph(
        [node_def],
        'Max_v9',
        [a, b, c],
        [d],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_max_case_v9.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_max_v9()
    make_max_v11()
    make_max_v12()
    make_max_v13()
