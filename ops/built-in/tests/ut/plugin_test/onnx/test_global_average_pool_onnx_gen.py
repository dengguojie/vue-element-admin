import onnx
from onnx import helper
from onnx import TensorProto


def make_global_average_pool_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 1, 1])
    node_def = helper.make_node(
        'GlobalAveragePool',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'GlobalAveragePool_v11',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_global_average_pool_case_v11.onnx")
    onnx.checker.check_model(model)


def make_global_average_pool_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 1, 1])
    node_def = helper.make_node(
        'GlobalAveragePool',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'GlobalAveragePool_v12',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_global_average_pool_case_v12.onnx")
    onnx.checker.check_model(model)


def make_global_average_pool_v13():
    """v13"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 1, 1])
    node_def = helper.make_node(
        'GlobalAveragePool',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'GlobalAveragePool_v13',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_global_average_pool_case_v13.onnx")
    onnx.checker.check_model(model)


def make_global_average_pool_v9():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 1, 1])
    node_def = helper.make_node(
        'GlobalAveragePool',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'GlobalAveragePool_v9',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_global_average_pool_case_v9.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_global_average_pool_v9()
    make_global_average_pool_v11()
    make_global_average_pool_v12()
    make_global_average_pool_v13()
