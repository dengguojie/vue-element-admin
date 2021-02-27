import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def export():
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.INT64, [3, ])

    node = onnx.helper.make_node(
        'Shape',
        inputs=['x'],
        outputs=['y'])

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[x],
        outputs=[y])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-Shape-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_Shape_case_V11.onnx")


if __name__ == "__main__":
    version_t = 11
    export()
