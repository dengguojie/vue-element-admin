import onnx
from onnx import helper
from onnx import TensorProto


def make_conv_transpose_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 5, 5])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_transpose_case_v11.onnx")
    onnx.checker.check_model(model)


def make_conv_transpose_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 5, 5])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_conv_transpose_case_v12.onnx")
    onnx.checker.check_model(model)


def make_conv_transpose_v13():
    """v13"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 5, 5])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_conv_transpose_case_v13.onnx")
    onnx.checker.check_model(model)


def make_conv_transpose_v9():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 5, 5])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_conv_transpose_case_v9.onnx")
    onnx.checker.check_model(model)

def make_conv_transpose_group():
    """v11 group"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 128, 16, 16])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT16, [128, 1, 4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1, 128, 32, 32])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
        strides=[2, 2],
        kernel_shape=[4, 4],
        dilations=[1, 1],
        pads=[1, 1, 1, 1],
        group=128
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_transpose_case_group.onnx")
    onnx.checker.check_model(model)

def make_conv_transpose_auto_pad():
    """v11 group"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 3, 32, 32])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT16, [3, 16, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1, 16, 66, 66])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
        strides=[2, 2],
        kernel_shape=[3, 3],
        dilations=[1, 1],
        auto_pad="SAME_LOWER",
        output_shape=[64, 64]
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_transpose_case_auto_pad.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_conv_transpose_v9()
    make_conv_transpose_v11()
    make_conv_transpose_v12()
    make_conv_transpose_v13()
    make_conv_transpose_group()
    make_conv_transpose_auto_pad()
