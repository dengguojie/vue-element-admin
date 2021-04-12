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


if __name__ == '__main__':
    random_uniform()
