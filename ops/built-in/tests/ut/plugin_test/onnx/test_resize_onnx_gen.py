import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np


def make_resize_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    roi = helper.make_tensor('roi', TensorProto.FLOAT, [1], [1])
    scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1, 1, 1, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', 'roi', 'scales'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        cubic_coeff_a = -0.75,
        exclude_outside = 0,
        extrapolation_value = 0,
        mode='linear',
        nearest_mode = "round_prefer_floor"
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

def make_resize_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    roi = helper.make_tensor('roi', TensorProto.FLOAT, [1], [1])
    scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1, 1, 1, 1])
    sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', 'roi', 'scales', 'sizes'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        mode='nearest'
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v12',
        [x, sizes],
        [y],
        [roi, scales],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_resize_case_v12.onnx")

def make_resize_v11_dong():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    roi = helper.make_tensor_value_info('roi', TensorProto.FLOAT, [1])
    scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [1, 1, 1, 1])
    sizes = helper.make_tensor('sizes', TensorProto.INT64, [4], [1, 1, 1, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', 'roi', 'scales', 'sizes'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        mode='linear'
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v11',
        [x, roi, scales],
        [y],
        [sizes],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_resize_case_v11_dong.onnx")

def make_resize_v12_dong():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    roi = helper.make_tensor_value_info('roi', TensorProto.FLOAT, [1])
    scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4,])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', 'roi', 'scales'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        mode='nearest'
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v11',
        [x, roi, scales],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_resize_case_v12_dong.onnx")

def make_resize_error():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 1, 1])
    roi = helper.make_tensor_value_info('roi', TensorProto.FLOAT, [1,])
    scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4,])
    sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4,])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', '', 'scales'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_pixel',
        mode='linerqa'
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v11',
        [x, roi, scales],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_resize_case_error.onnx")


def make_resize_v10():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1, 1, 1, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', 'scales'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        cubic_coeff_a = -0.75,
        exclude_outside = 0,
        extrapolation_value = 0,
        mode='linear',
        nearest_mode = "round_prefer_floor"
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v10',
        [x],
        [y],
        [scales]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 10
    onnx.save(model, "./test_resize_case_v10.onnx")

def make_resize_v10_1():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1, 1, 1, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Resize',
        inputs=['x', 'scales'],
        outputs=['y'],
        coordinate_transformation_mode='pytorch_half_pixel',
        mode='nearest'
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v10',
        [x],
        [y],
        [scales]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 10
    onnx.save(model, "./test_resize_case_v10_1.onnx")

if __name__ == '__main__':
    make_resize_v11()
    make_resize_v12()
    make_resize_v11_dong()
    make_resize_v12_dong()
    make_resize_error()
    make_resize_v10_1()
    make_resize_v10()
