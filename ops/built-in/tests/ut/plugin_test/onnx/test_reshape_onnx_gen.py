import onnx
from onnx import helper

def reshape():
    node = helper.make_node('Reshape',
        inputs=['data', 'shape'],
        outputs=['reshaped'],
        allowzero=0,
        name='test_reshaped_1')
    shape = helper.make_tensor(
            "shape", onnx.TensorProto.INT64, [3], [4, 2, 3])
    
    graph = helper.make_graph(
        [node],
        "test_reshaped_1",
        [helper.make_tensor_value_info(
            "data", onnx.TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info(
            "reshaped", onnx.TensorProto.FLOAT, [4, 2, 3])],
        [shape]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_reshape_case_1.onnx")


def make_reshape_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [5, 4, 3, 2])
    shape = helper.make_tensor('shape', onnx.TensorProto.INT64, [2, ], [5, 24])
    y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [5, 24])
    node_def = helper.make_node(
        'Reshape',
        inputs=['x', 'shape'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Reshape_v12',
        [x],
        [y],
        [shape]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_reshape_case_v12.onnx")
    onnx.checker.check_model(model)


def make_reshape_v13():
    """v13"""
    x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [5, 4, 3, 2])
    shape = helper.make_tensor('shape', onnx.TensorProto.INT64, [2, ], [5, 24])
    y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [5, 24])
    node_def = helper.make_node(
        'Reshape',
        inputs=['x', 'shape'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Reshape_v13',
        [x],
        [y],
        [shape]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_reshape_case_v13.onnx")
    onnx.checker.check_model(model)


def make_reshape_v9():
    """v9"""
    x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [5, 4, 3, 2])
    shape = helper.make_tensor('shape', onnx.TensorProto.INT64, [2, ], [5, 24])
    y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [5, 24])
    node_def = helper.make_node(
        'Reshape',
        inputs=['x', 'shape'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'Reshape_v9',
        [x],
        [y],
        [shape]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_reshape_case_v9.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    reshape()
    make_reshape_v9()
    make_reshape_v12()
    make_reshape_v13()
