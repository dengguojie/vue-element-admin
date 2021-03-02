import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def export_reverseSequence_1d(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, ])
    sequence_lens = helper.make_tensor('sequence_lens', TensorProto.FLOAT, [4, ], [4, 3, 2, 1])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, ])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x', 'sequence_lens'],
        outputs=['y'],
        time_axis=0,
        batch_axis=1,)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [sequence_lens])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReverseSequence-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_ReverseSequence_onnx_V11case1.onnx")


def reverseSequence_1d_exclusive(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, ])
    sequence_lens = helper.make_tensor('sequence_lens', TensorProto.FLOAT, [4, ], [1, 2, 3, 4])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, ])

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x', 'sequence_lens'],
        outputs=['y'],
        time_axis=1,
        batch_axis=0,)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [sequence_lens])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReverseSequence-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_ReverseSequence_onnx_V11case2.onnx")


if __name__ == "__main__":
    version = 11
    export_reverseSequence_1d(version)
    reverseSequence_1d_exclusive(version)
