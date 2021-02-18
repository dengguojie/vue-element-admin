#abs.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

def Abs():
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 4, 5])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 4, 5])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'Abs',
        inputs=['input'],
        outputs=['output']
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-Abs-onnx')
    model_def.opset_import[0].version = 12  # 指定12版本

    #onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./test_abs_case.onnx")        # 保存onnx模型
    print('The model is:\n{}'.format(model_def))

if __name__ == "__main__":
    Abs()
