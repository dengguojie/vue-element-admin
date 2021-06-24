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

def make_conv3d_transpose_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 320, 7, 7, 5])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [320, 320, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 320, 14, 14, 10])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
        dilations=[1, 1, 1],
        group=1,
        kernel_shape=[2, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
        strides=[2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],

    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_conv3d_transpose_case_v11.onnx")
    onnx.checker.check_model(model)

def make_conv3d_transpose_v12():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 320, 7, 7, 5])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [320, 320, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 320, 14, 14, 10])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
        dilations=[1, 1, 1],
        group=1,
        kernel_shape=[2, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
        strides=[2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],

    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_conv3d_transpose_case_v12.onnx")
    onnx.checker.check_model(model)

def make_conv3d_transpose_v13():
    """v13"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 320, 7, 7, 5])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [320, 320, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 320, 14, 14, 10])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
        dilations=[1, 1, 1],
        group=1,
        kernel_shape=[2, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
        strides=[2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],

    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_conv3d_transpose_case_v13.onnx")
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

def make_conv_transpose_v8():
    """v8"""
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
    model.opset_import[0].version = 8
    onnx.save(model, "./test_conv_transpose_case_v8.onnx")
    onnx.checker.check_model(model)

def make_conv3d_transpose_v8():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 320, 7, 7, 5])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [320, 320, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 320, 14, 14, 10])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
        dilations=[1, 1, 1],
        group=1,
        kernel_shape=[2, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
        strides=[2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],

    )
    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_conv3d_transpose_case_v8.onnx")
    onnx.checker.check_model(model)


def make_conv3d_transpose_v9():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 320, 7, 7, 5])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [320, 320, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 320, 14, 14, 10])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
        dilations=[1, 1, 1],
        group=1,
        kernel_shape=[2, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
        strides=[2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],

    )
    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_conv3d_transpose_case_v9.onnx")
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

def make_conv3d_transpose_auto_pad():
    """v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 320, 7, 7, 5])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [320, 320, 2, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 320, 14, 14, 10])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'w'],
        outputs=['y'],
        dilations=[1, 1, 1],
        group=1,
        kernel_shape=[2, 2, 2],
        auto_pad="SAME_LOWER",
        strides=[2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],

    )
    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_conv3d_transpose_case_auto_pad.onnx")
    onnx.checker.check_model(model)

def make_conv2d_Transpose_3_inputs():
    """v11 2d 3-input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1024, 35, 35])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1024, 512, 2, 2])
    b = helper.make_tensor_value_info('B', TensorProto.FLOAT, [512, ])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 512, 70, 70])
    node_def = helper.make_node(
        'ConvTranspose',
        inputs=['x', 'W', 'B'],
        outputs=['y'],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[2, 2],
        dilations=[1, 1],
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
    onnx.save(model, "./test_conv2d_transpose_case_3_input.onnx")


if __name__ == '__main__':
    make_conv_transpose_v8()
    make_conv3d_transpose_v8()
    make_conv_transpose_v9()
    make_conv3d_transpose_v9()
    make_conv_transpose_v11()
    make_conv3d_transpose_v11()
    make_conv_transpose_v12()
    make_conv3d_transpose_v12()
    make_conv_transpose_v13()
    make_conv3d_transpose_v13()
    make_conv_transpose_group()
    make_conv_transpose_auto_pad()
    make_conv3d_transpose_auto_pad()
    make_conv2d_Transpose_3_inputs()
