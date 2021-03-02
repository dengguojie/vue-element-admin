import onnx
from onnx import helper
from onnx import TensorProto


def make_batch_normalization_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 1, 3])
    s = helper.make_tensor_value_info('s', TensorProto.FLOAT, [2, ])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, ])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, ])
    var = helper.make_tensor_value_info('var', TensorProto.FLOAT, [2, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 1, 3])
    node_def = helper.make_node(
        'BatchNormalization',
        inputs=['x', 's', 'bias', 'mean', 'var'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'BatchNormalization_v11',
        [x, s, bias, mean, var],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_batch_normalization_case_v11.onnx")
    onnx.checker.check_model(model)


def make_batch_normalization_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 1, 3])
    s = helper.make_tensor_value_info('s', TensorProto.FLOAT, [2, ])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, ])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, ])
    var = helper.make_tensor_value_info('var', TensorProto.FLOAT, [2, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 1, 3])
    node_def = helper.make_node(
        'BatchNormalization',
        inputs=['x', 's', 'bias', 'mean', 'var'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'BatchNormalization_v12',
        [x, s, bias, mean, var],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_batch_normalization_case_v12.onnx")
    onnx.checker.check_model(model)


def make_batch_normalization_v13():
    """v13"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 1, 3])
    s = helper.make_tensor_value_info('s', TensorProto.FLOAT, [2, ])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, ])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, ])
    var = helper.make_tensor_value_info('var', TensorProto.FLOAT, [2, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 1, 3])
    node_def = helper.make_node(
        'BatchNormalization',
        inputs=['x', 's', 'bias', 'mean', 'var'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'BatchNormalization_v13',
        [x, s, bias, mean, var],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_batch_normalization_case_v13.onnx")
    onnx.checker.check_model(model)


def make_batch_normalization_v9():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 1, 3])
    s = helper.make_tensor_value_info('s', TensorProto.FLOAT, [2, ])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2, ])
    mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2, ])
    var = helper.make_tensor_value_info('var', TensorProto.FLOAT, [2, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 1, 3])
    node_def = helper.make_node(
        'BatchNormalization',
        inputs=['x', 's', 'bias', 'mean', 'var'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'BatchNormalization_v9',
        [x, s, bias, mean, var],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_batch_normalization_case_v9.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_batch_normalization_v9()
    make_batch_normalization_v11()
    make_batch_normalization_v12()
    make_batch_normalization_v13()
