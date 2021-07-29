import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as rt
import numpy as np


def hardswish():
    """v11"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 5])
    node_def = helper.make_node(
        'HardSwish',
        inputs=['x'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'HardSwish_v14',
        [x],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 14
    onnx.save(model, "./test_hardswish_case_v14.onnx")

if __name__ == '__main__':
    hardswish()
