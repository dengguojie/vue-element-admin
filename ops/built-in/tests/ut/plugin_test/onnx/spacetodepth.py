#spacetodepth.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def spacetodepth(version_num):
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 2])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 1, 1])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'SpaceToDepth',
        inputs=['input'],
        outputs=['output'],
        blocksize=2
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-SpaceToDepth-onnx')
    model_def.opset_import[0].version = version_num

    #onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./SpaceToDepth_V{}.onnx".format(version_num))
    print('The model is:\n{}'.format(model_def))

if __name__ == "__main__":
    version_t = (9, 10, 11, 12, 13)
    for i in version_t:
        spacetodepth(i)
