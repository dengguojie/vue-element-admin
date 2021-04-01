import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def export_default_axes_keepdims(V):

    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [1, 1, 1])
    
    node = onnx.helper.make_node(
        'LpNormalization',
        ['data'],
        ['reduced'],
        axis=-1,
        p=2
        )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [data],
        [reduced])

    model_def = helper.make_model(graph_def, producer_name='HJ-LpNormalization-onnx')
    model_def.opset_import[0].version = V  # version

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_LpNormalization_case1.onnx")

if __name__ == "__main__":
    version_t = 9
    export_default_axes_keepdims(version_t)
