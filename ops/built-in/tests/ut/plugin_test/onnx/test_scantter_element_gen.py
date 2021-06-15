import onnx
from onnx import helper
from onnx import TensorProto


def make_scantt_v11():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1,5])
    roi = helper.make_tensor_value_info('roi', TensorProto.INT64, [1,2])
    scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [1,2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'ScatterElements',
        inputs=['x', 'roi', 'scales'],
        outputs=['y'],
        axis=1,
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v11',
        [x, roi, scales],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_scatter_element_case_v11.onnx")
    onnx.checker.check_model(model)


def make_scantt_v11_1():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1,5])
    roi = helper.make_tensor_value_info('roi', TensorProto.INT64, [1,2])
    scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [1,2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'ScatterElements',
        inputs=['x', 'roi', 'scales'],
        outputs=['y'],
        axis=-1,
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v11',
        [x, roi, scales],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_scatter_element_case_v11_1.onnx")
    onnx.checker.check_model(model)

def make_scantt_v11_2():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1,5])
    roi = helper.make_tensor_value_info('roi', TensorProto.INT64, [1,2])
    scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [1,2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    node_def = helper.make_node(
        'Scatter',
        inputs=['x', 'roi', 'scales'],
        outputs=['y'],
        axis=1,
    )

    graph = helper.make_graph(
        [node_def],
        'Resize_v11',
        [x, roi, scales],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_scatter_case_v11.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_scantt_v11()
    make_scantt_v11_1()
    make_scantt_v11_2()
