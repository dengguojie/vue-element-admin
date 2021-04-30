import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def export_elu():
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, ])
  
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, ])

    node = onnx.helper.make_node(
        'Elu',
        inputs=['x'],
        outputs=['y'],
        alpha=1.0)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = 11

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_elu_onnx_case1.onnx")

if __name__ == '__main__':
    export_elu()