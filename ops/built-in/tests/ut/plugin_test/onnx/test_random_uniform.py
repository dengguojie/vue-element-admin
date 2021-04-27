import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def random_uniform():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [10])

    # Create two node
    random = helper.make_node(
        'RandomUniform',
        ['random_x'],
        ['random_y'],
        shape=[10],
        high=100.0,
        low=0.0,
    )

    add = helper.make_node(
        'Add',  # node name
        ['x', 'random_y'],  # inputs
        ['Y'],  # outputs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [random, add],
        'randomuniform',
        inputs=[x],
        outputs=[Y],
    )

    model_def = helper.make_model(graph_def, producer_name='random-onnx')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_random_uniform.onnx")

def random_int():
    # Create two input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.INT32, [10])
    random = helper.make_tensor_value_info("random", TensorProto.INT32, [10])

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.INT32, [10])

    # Create two node
    node_1 = helper.make_node(
        'RandomUniform', # node name
        [], # inputs
        ['random'], # outputs
        shape=[10],
        dtype=6,
        high=99.0,
        low=95.0,
    )

    node_def = helper.make_node(
        'Add', # node name
        ['x', 'random'], # inputs
        ['Y'], # outputs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def, node_1],
        'test-model',
        inputs=[x],
        outputs=[Y],
    )

    model_def = helper.make_model(graph_def, producer_name='zyw-random-onnx')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./random_INT.onnx")


if __name__ == '__main__':
    random_uniform()
    random_int()
