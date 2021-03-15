import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

def gen_gloabl_max_pool_case1():
    #Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 6, 6])

    #Create one output (ValueOutputProto)
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 1, 1])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'GlobalMaxPool',
        inputs=['input'],
        outputs=['output']
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
    onnx.save(model_def, "./test_global_max_pool_case1.onnx")
    print('The model is:\n{}'.format(model_def))

if __name__ == "__main__":
    gen_gloabl_max_pool_case1()
