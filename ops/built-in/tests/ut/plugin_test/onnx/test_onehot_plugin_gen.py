import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

def onehot():
    #Create one input (ValueInfoProto)
    indices = helper.make_tensor_value_info('indices', TensorProto.FLOAT, [2, 2])
    depth = helper.make_tensor('depth', TensorProto.FLOAT, [1], [10])
    values = helper.make_tensor_value_info('values', TensorProto.FLOAT, [2])

    #Create one output (ValueOutputProto)
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 4, 4])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'OneHot',
        inputs=['indices', 'depth', 'values'],
        outputs=['output'],
        axis=1,
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [indices, values],
        [y],
        [depth]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onehot-onnx')
    model_def.opset_import[0].version = 11

    #onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./test_onehot_case.onnx")
    print('The model is:\n{}'.format(model_def))

if __name__ == "__main__":
    onehot()
