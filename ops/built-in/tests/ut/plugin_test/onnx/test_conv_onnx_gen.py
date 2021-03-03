import onnx
from onnx import helper
from onnx import TensorProto


def make_conv_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1]
    )

    graph = helper.make_graph(
        [node_def],
        'Conv_v11',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_case_v11.onnx")


def make_conv_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1]
    )

    graph = helper.make_graph(
        [node_def],
        'Conv_v12',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_conv_case_v12.onnx")


def make_conv_v13():
    """v13"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1]
    )

    graph = helper.make_graph(
        [node_def],
        'Conv_v13',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_conv_case_v13.onnx")


def make_conv_v9():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1]
    )

    graph = helper.make_graph(
        [node_def],
        'Conv_v9',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_conv_case_v9.onnx")


if __name__ == '__main__':
    make_conv_v9()
    make_conv_v11()
    make_conv_v12()
    make_conv_v13()
