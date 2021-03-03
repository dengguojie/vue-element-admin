#mvn_v2.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

def mvn_v2():
    #Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 4, 4])

    #Create one output (ValueOutputProto)
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 4, 4])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'MeanVarianceNormalization',
        inputs=['input'],
        outputs=['output'],
        axes=[2, 3]
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [x],
        [y]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='mvnv2-onnx')
    model_def.opset_import[0].version = 11

    #onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./test_mvn_v2_case.onnx")
    print('The model is:\n{}'.format(model_def))

if __name__ == "__main__":
    mvn_v2()
