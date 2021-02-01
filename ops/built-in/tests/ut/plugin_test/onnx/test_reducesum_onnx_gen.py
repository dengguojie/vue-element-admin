import onnx
from onnx import helper, TensorProto


def default_axes_keepdims():
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info(
        'reduced', TensorProto.FLOAT, [3, 2, 2])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        'ReduceSum',  # node name
        inputs=['data'],  # inputs
        outputs=['reduced'],  # outputs
        keepdims=1,  # attributes
        axes=[0, 1, 2]
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        inputs=[data],
        outputs=[reduced],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = 11  # version 11

    onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./reduce_sum_default_axes_keepdims.onnx")  # save onnx model
    print('The model is:\n{}'.format(model_def))


def do_not_keepdims():
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])
    # axes = helper.make_tensor_value_info('axes', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info(
        'reduced', TensorProto.FLOAT, [3, 2, 2])

    # axes = helper.make_tensor_value_info('axes', AttributeProto.INT, [])
    # keepdims = helper.make_tensor_value_info('keepdims', AttributeProto.INT, [])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        'ReduceSum',  # node name
        inputs=['data'],  # inputs
        outputs=['reduced'],  # outputs
        keepdims=0,  # alpha=2.0, # attributes
        axes=[1]
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        inputs=[data],
        outputs=[reduced],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = 11  # version 11

    onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./reduce_sum_do_not_keepdims.onnx")  # save onnx model
    print('The model is:\n{}'.format(model_def))


def keepdims():
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])
    # axes = helper.make_tensor_value_info('axes', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info(
        'reduced', TensorProto.FLOAT, [3, 2, 2])

    # axes = helper.make_tensor_value_info('axes', AttributeProto.INT, [])
    # keepdims = helper.make_tensor_value_info('keepdims', AttributeProto.INT, [])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        'ReduceSum',  # node name
        inputs=['data'],  # inputs
        outputs=['reduced'],  # outputs
        keepdims=1,  # alpha=2.0, # attributes
        axes=[1]
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        inputs=[data],
        outputs=[reduced],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = 11  # version 11

    onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./reduce_sum_keepdims.onnx")  # save onnx model
    print('The model is:\n{}'.format(model_def))


def negative_axes_keepdims():
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])
    # axes = helper.make_tensor_value_info('axes', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info(
        'reduced', TensorProto.FLOAT, [3, 2, 2])

    # axes = helper.make_tensor_value_info('axes', AttributeProto.INT, [])
    # keepdims = helper.make_tensor_value_info('keepdims', AttributeProto.INT, [])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        'ReduceSum',  # node name
        inputs=['data'],  # inputs
        outputs=['reduced'],  # outputs
        keepdims=1,  # alpha=2.0, # attributes
        axes=[-2]
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        inputs=[data],
        outputs=[reduced],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = 11  # version 11

    onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./reduce_sum_negative_axes_keepdims.onnx")  # save onnx model
    print('The model is:\n{}'.format(model_def))


def default_axes_keepdims_int64():
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.INT64, [3, 2, 2])  # change to TensorProto.INT64

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info(
        'reduced', TensorProto.FLOAT, [3, 2, 2])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        'ReduceSum',  # node name
        inputs=['data'],  # inputs
        outputs=['reduced'],  # outputs
        keepdims=1,  # attributes
        axes=[0, 1, 2]
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        inputs=[data],
        outputs=[reduced]
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = 11  # version 11

    onnx.checker.check_model(model_def)
    print('The model is checked!')
    onnx.save(model_def, "./reduce_sum_default_axes_keepdims_int64.onnx")  # save onnx model
    print('The model is:\n{}'.format(model_def))


if __name__ == '__main__':
    default_axes_keepdims()
    do_not_keepdims()
    keepdims()
    negative_axes_keepdims()
    default_axes_keepdims_int64()
