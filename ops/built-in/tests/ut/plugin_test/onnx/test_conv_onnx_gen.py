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


def make_conv_2d_3_inputs():
    """v11 2d 3-input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    w = helper.make_tensor('W', TensorProto.FLOAT, [1, 1, 3, 3], [
                           1, 1, 1, 1, 1, 1, 1, 1, 1])
    b = helper.make_tensor('B', TensorProto.FLOAT, [1, ], [1, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W', 'B'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 1],
        group=1
    )

    graph = helper.make_graph(
        [node_def],
        'Conv',
        [x],
        [y],
        [w, b]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_case_2d_3_input.onnx")


def make_conv_3d_2_inputs():
    """v11 3d 2-input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 7, 7, 7])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [64, 3, 7, 7, 7])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 64, 7, 4, 4])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[7, 7, 7],
        pads=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2],
        dilations=[1, 1, 1],
        group=1
    )

    graph = helper.make_graph(
        [node_def],
        'Conv',
        [x, w],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_case_3d_2_input.onnx")


def make_conv_3d_3_inputs():
    """v11 3d 3-input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 7, 7, 7])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [64, 3, 7, 7, 7])
    b = helper.make_tensor_value_info('B', TensorProto.FLOAT, [64, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 64, 7, 4, 4])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W', 'B'],
        outputs=['y'],
        kernel_shape=[7, 7, 7],
        pads=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2],
        dilations=[1, 1, 1],
        group=1
    )

    graph = helper.make_graph(
        [node_def],
        'Conv',
        [x, w, b],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_case_3d_3_input.onnx")


def make_conv_1d_2_inputs():
    """v11 1d 2-input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 16, 50])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [33, 16, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 33, 24])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        strides=[2, ],
        kernel_shape=[2, ],
        dilations=[2, ],
        pads=[1, 1],
    )

    graph = helper.make_graph(
        [node_def],
        'Conv',
        [x, w],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_case_1d_2_input.onnx")

def make_conv_1d_inputs():
    """v11 1d 2-input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 16, 50])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [33, 16, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 33, 24])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        strides=[2, ],
        dilations=[2, ],
        pads=[1, 1],
    )

    graph = helper.make_graph(
        [node_def],
        'Conv',
        [x, w],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_case_1d_input.onnx")

def make_conv_autopad_same():
    """v11 2d 2-input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        auto_pad="SAME_LOWER",
        kernel_shape = [2, 2],
    )

    graph = helper.make_graph(
        [node_def],
        'Conv',
        [x, w],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_case_with_autopad_same.onnx")

def make_conv_fail():
    """v11 2d 2-input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])
    node_def = helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Conv',
        [x, w],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv_case_fail.onnx")


if __name__ == '__main__':
    make_conv_v9()
    make_conv_v11()
    make_conv_v12()
    make_conv_v13()
    make_conv_2d_3_inputs()
    make_conv_3d_2_inputs()
    make_conv_3d_3_inputs()
    make_conv_1d_2_inputs()
    make_conv_autopad_same()
    make_conv_fail()
    make_conv_1d_inputs()
