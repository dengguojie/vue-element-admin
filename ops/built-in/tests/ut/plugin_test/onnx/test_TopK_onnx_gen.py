import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

def top_k():
    axis = 1

    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 4])
    k = helper.make_tensor('K', TensorProto.INT64, [1], [3])
    # Create one output (ValueInfoProto)
    values_ref = helper.make_tensor_value_info('values_ref', TensorProto.INT64, [3, 3])
    indices_ref = helper.make_tensor_value_info('indices_ref', TensorProto.INT64, [3, 3])

    node = onnx.helper.make_node(
        'TopK',
        inputs=['X', 'K'],
        outputs=['values_ref', 'indices_ref'],
        axis=axis)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [values_ref, indices_ref],
        [k])

    model_def = helper.make_model(graph_def, producer_name='HJ-TopKV11-onnx')
    model_def.opset_import[0].version = 11

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./TopK_V11_case.onnx")

def top_k_fail_v9():
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 4])
    # k = helper.make_tensor('k', TensorProto.FLOAT, [1, ], [3])
    # Create one output (ValueInfoProto)
    values_ref = helper.make_tensor_value_info('values', TensorProto.INT32, [3, 3])
    indices_ref = helper.make_tensor_value_info('indices', TensorProto.INT32, [3, 3])

    node = onnx.helper.make_node(
        'TopK',
        inputs=['X'],
        outputs=['values', 'indices'],
        k=-1)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [values_ref, indices_ref])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='HJ-TopKV9-onnx')
    model_def.opset_import[0].version = 9

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./TopK_V9__fail_case.onnx")

def top_k_v9():
    axis = 1
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 3])
    # Create one output (ValueInfoProto)
    values_ref = helper.make_tensor_value_info('values_ref', TensorProto.FLOAT, [3, 3])
    indices_ref = helper.make_tensor_value_info('indices_ref', TensorProto.INT32, [3, 3])

    node = onnx.helper.make_node(
        'TopK',
        inputs=['X'],
        outputs=['values_ref', 'indices_ref'],
        k=3,
        axis=axis)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [values_ref, indices_ref])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='HJ-TopK-onnx')
    model_def.opset_import[0].version = 9

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./TopK_V9_case.onnx")

if __name__ == "__main__":
    top_k()
    top_k_fail_v9()
    top_k_v9()
