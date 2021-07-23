import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def isinf_error():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6,])
    y = helper.make_tensor_value_info("y", TensorProto.BOOL, [6,])
    node_def = helper.make_node('IsInf',
                                inputs=['x'],
                                outputs=['y'],
                                detect_positive=0
                                )
    graph = helper.make_graph(
        [node_def],
        "test_isinf_case_1",
        [x],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-isinf_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_isinf_error_v11.onnx")

def isinf_default():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6,])
    y = helper.make_tensor_value_info("y", TensorProto.BOOL, [6,])
    node_def = helper.make_node('IsInf',
                                inputs=['x'],
                                outputs=['y'],
                                )
    graph = helper.make_graph(
        [node_def],
        "test_isinf_case_1",
        [x],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-isinf_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_isinf_v11.onnx")

if __name__ == "__main__":
    isinf_error()
    isinf_default()