import onnx
from onnx import helper
from onnx import TensorProto


def make_global():
    """same shape"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 6])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 2, 6])
    node_def = helper.make_node(
        'GlobalLpPool',
        inputs=['x'],
        outputs=['z'],
    )

    graph = helper.make_graph(
        [node_def],
        'GlobalLpPool',
        [x],
        [z],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_global_lppool_case_1.onnx")
    onnx.checker.check_model(model)


def make_global_1():
    """same shape"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2, 6])
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 2, 6])
    node_def = helper.make_node(
        'GlobalLpPool',
        inputs=['x'],
        outputs=['z'],
        p=3,
    )

    graph = helper.make_graph(
        [node_def],
        'GlobalLpPool',
        [x],
        [z],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_global_lppool_case_2.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_global()
    make_global_1()
