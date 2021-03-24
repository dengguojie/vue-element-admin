import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def export(version_t):
    # Create one input (ValueInfoProto)

    data_0 = helper.make_tensor_value_info('data_0', TensorProto.FLOAT, [3, ])
    data_1 = helper.make_tensor_value_info('data_1', TensorProto.FLOAT, [3, ])
    data_2 = helper.make_tensor_value_info('data_2', TensorProto.FLOAT, [3, ])
    # Create one output (ValueInfoProto)
    result = helper.make_tensor_value_info('result', TensorProto.FLOAT, [3, ])

    node = onnx.helper.make_node(
        'Mean',
        inputs=['data_0', 'data_1', 'data_2'],
        outputs=['result'],)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data_0, data_1, data_2],
        outputs=[result]
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='HJ-Mean-onnx')
    model_def.opset_import[0].version = version_t  # version

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_Mean_case1.onnx")


def export_one(version_t):
    data_0 = helper.make_tensor_value_info('data_0', TensorProto.FLOAT, [3, ])
    result = helper.make_tensor_value_info('result', TensorProto.FLOAT, [3, ])
    
    node = onnx.helper.make_node(
        'Mean',
        inputs=['data_0'],
        outputs=['result'],)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data_0],
        outputs=[result]
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='HJ-Mean-onnx')
    model_def.opset_import[0].version = version_t  # version

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_Mean_case2.onnx")


if __name__ == "__main__":
    version_v = 9
    export(version_v)
    export_one(version_v)
