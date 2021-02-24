#reduceL2.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def reduceL2_default():
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 2])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 1])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'ReduceL2',
        inputs=['input'],
        outputs=['output'],
        keepdims=1
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceL2-onnx')
    model_def.opset_import[0].version = version_num

    #onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./ReduceL2_default.onnx")
    print('The model is:\n{}'.format(model_def))


def reduceL2_do_not_keepdims():
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 2])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 2])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'ReduceL2',
        inputs=['input'],
        outputs=['output'],
        axes = [2],
        keepdims = 0
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceL2-onnx')
    model_def.opset_import[0].version = version_num

    #onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./ReduceL2_not_keepdims.onnx")
    print('The model is:\n{}'.format(model_def))


def reduceL2_keepdims():
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 2])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 2, 1])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'ReduceL2',
        inputs=['input'],
        outputs=['output'],
        axes = [2],
        keepdims = 1
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceL2-onnx')
    model_def.opset_import[0].version = version_num

    #onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./ReduceL2_keepdims.onnx")
    print('The model is:\n{}'.format(model_def))


def reduceL2_negative_axes_keepdims():
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 2])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 2, 1])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'ReduceL2',
        inputs=['input'],
        outputs=['output'],
        axes = [-1],
        keepdims = 1
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceL2-onnx')
    model_def.opset_import[0].version = version_num

    #onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./ReduceL2_negative.onnx")
    print('The model is:\n{}'.format(model_def))

if __name__ == "__main__":
    version_num = 11
    reduceL2_default()
    reduceL2_do_not_keepdims()
    reduceL2_keepdims()
    reduceL2_negative_axes_keepdims()
