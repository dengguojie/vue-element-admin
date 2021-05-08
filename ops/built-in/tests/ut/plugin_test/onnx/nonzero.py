#nonzero.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def nonzero():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('x', TensorProto.BOOL, [2, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('y', TensorProto.INT64, [2, 3])

    node_def = helper.make_node(
    'NonZero', # node name
    inputs=['x'], # inputs
    outputs=['y'], # outputs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyw-onnx-nonzero')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./onnx/test_nonzero.onnx")

if __name__ == "__main__":
    nonzero()
