import onnx
from onnx import helper
from onnx import TensorProto


def make_resize_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    roi = helper.make_tensor('roi', TensorProto.FLOAT, [1], [0.1])
    scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1, 1, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', 'roi', 'scales'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        mode='linear'
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v11',
        [x],
        [y],
        [roi, scales]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_resize_case_v11.onnx")
    onnx.checker.check_model(model)


def make_resize_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    roi = helper.make_tensor('roi', TensorProto.FLOAT, [1], [0.1])
    scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1, 1, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', 'roi', 'scales'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        mode='linear'
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v12',
        [x],
        [y],
        [roi, scales]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_resize_case_v12.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_resize_v11()
    make_resize_v12()
