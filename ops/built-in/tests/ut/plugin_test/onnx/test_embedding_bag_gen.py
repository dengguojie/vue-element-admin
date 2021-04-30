import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def export_bag():
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, ])
  
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, ])

    node = onnx.helper.make_node(
        'EmbeddingBag',
        inputs=['x'],
        outputs=['y'],
        mode="auto",
        scale_grad_by_freq=1,
        sparse=1,
        include_last_offset=1)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = 11
    
    onnx.save(model_def, "./test_embedding_bag_onnx_case1.onnx")
    onnx.checker.check_model(model_def)

if __name__ == '__main__':
    export_bag()