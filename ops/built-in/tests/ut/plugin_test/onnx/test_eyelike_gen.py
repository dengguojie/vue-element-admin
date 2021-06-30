import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def eyelike():
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 5])
  
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 5])

    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['x'],
        outputs=['y'],
        dtype=3,
        k=0,
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = 11
    
    onnx.save(model_def, "./test_eyelike_onnx_case1.onnx")
    onnx.checker.check_model(model_def)

if __name__ == '__main__':
    eyelike()