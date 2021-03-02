import onnx
from onnx import helper
from onnx import TensorProto


def make_sum_v11():
    """v11"""
    x_1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, ])
    x_2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, ])
    x_3 = helper.make_tensor_value_info('x3', TensorProto.FLOAT, [3, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, ])
    node_def = helper.make_node(
        'Sum',
        inputs=['x1', 'x2', 'x3'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Sum_v11',
        [x_1, x_2, x_3],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_sum_case_v11.onnx")
    onnx.checker.check_model(model)


def make_sum_v12():
    """v12"""
    x_1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, ])
    x_2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, ])
    x_3 = helper.make_tensor_value_info('x3', TensorProto.FLOAT, [3, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, ])
    node_def = helper.make_node(
        'Sum',
        inputs=['x1', 'x2', 'x3'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Sum_v12',
        [x_1, x_2, x_3],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_sum_case_v12.onnx")
    onnx.checker.check_model(model)


def make_sum_v13():
    """v13"""
    x_1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, ])
    x_2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, ])
    x_3 = helper.make_tensor_value_info('x3', TensorProto.FLOAT, [3, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, ])
    node_def = helper.make_node(
        'Sum',
        inputs=['x1', 'x2', 'x3'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Sum_v13',
        [x_1, x_2, x_3],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_sum_case_v13.onnx")
    onnx.checker.check_model(model)


def make_sum_v9():
    """v9"""
    x_1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3, ])
    x_2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3, ])
    x_3 = helper.make_tensor_value_info('x3', TensorProto.FLOAT, [3, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, ])
    node_def = helper.make_node(
        'Sum',
        inputs=['x1', 'x2', 'x3'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Sum_v9',
        [x_1, x_2, x_3],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_sum_case_v9.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_sum_v9()
    make_sum_v11()
    make_sum_v12()
    make_sum_v13()
