import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def isnan_default(i):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4,])
    y = helper.make_tensor_value_info("y", TensorProto.BOOL, [4,])
    node_def = helper.make_node('IsNaN',
                                inputs=['x'],
                                outputs=['y'],
                                )
    graph = helper.make_graph(
        [node_def],
        "test_isnan_case_1",
        [x],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-isnan_test")
    model.opset_import[0].version = i
    onnx.save(model, "./onnx/test_isnan_v{}.onnx".format(i))

if __name__ == "__main__":
    version_t = (9, 10, 11, 12, 13)
    for i in version_t:
        isnan_default(i)
