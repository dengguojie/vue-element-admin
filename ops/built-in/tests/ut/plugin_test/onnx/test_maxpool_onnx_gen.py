import onnx
from onnx import helper
from onnx import TensorProto


def make_max_pool():
    """2d default"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 31, 31])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_maxpool_case_1.onnx")
    onnx.checker.check_model(model)


def make_max_pool1():
    """2d pads"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 28, 28])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 30, 30])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[2, 2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_maxpool_case_2.onnx")
    onnx.checker.check_model(model)


def make_max_pool2():
    """2d precomputed same upper"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 3, 3])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        auto_pad="SAME_UPPER"
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_maxpool_case_3.onnx")
    onnx.checker.check_model(model)


def make_max_pool3():
    """2d ceil"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 2])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        ceil_mode=True
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_maxpool_case_4.onnx")
    onnx.checker.check_model(model)


def make_max_pool4():
    """2d strides"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 10, 10])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[5, 5],
        strides=[3, 3]
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_maxpool_case_5.onnx")
    onnx.checker.check_model(model)


def make_max_pool_v12():
    """2d pads v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 28, 28])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 30, 30])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[2, 2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool_v12',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_maxpool_case_v12.onnx")
    onnx.checker.check_model(model)


def make_max_pool_v13():
    """2d pads v13"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 28, 28])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 30, 30])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[2, 2, 2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool_v13',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_maxpool_case_v13.onnx")
    onnx.checker.check_model(model)


def make_max_pool_v9():
    """2d default v9"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 31, 31])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, 2]
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool_v9',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_maxpool_case_v9.onnx")
    onnx.checker.check_model(model)


def make_max_pool_aicpu():
    """aicpu op"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 150, 150])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 3, 3])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[50, 50],
        strides=[50, 50],
        auto_pad="NOTSET",
        dilations=[1, 1]
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_maxpool_case_aicpu.onnx")
    onnx.checker.check_model(model)


def make_max_pool_1d():
    """1d input"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 32])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 31])
    node_def = helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, ],
        strides=[1, ],
        auto_pad="NOTSET",
        dilations=[1, ],
        pads=[0, 0]
    )

    graph = helper.make_graph(
        [node_def],
        'MaxPool',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_maxpool_case_1d_input.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_max_pool()
    make_max_pool1()
    make_max_pool2()
    make_max_pool3()
    make_max_pool4()
    make_max_pool_v9()
    make_max_pool_v12()
    make_max_pool_v13()
    make_max_pool_aicpu()
    make_max_pool_1d()
