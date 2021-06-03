import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def random_uniform_like():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3,4,5])
    Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3,4,5])

    # Create two node
    random = helper.make_node(
        'RandomUniformLike',
        ['x'],
        ['y'],
        dtype=1,
        low=0.5,
        high=1.5,
        seed=0.1,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [random],
        'randomuniform',
        inputs=[x],
        outputs=[Y],
    )

    model_def = helper.make_model(graph_def, producer_name='random-onnx')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_random_uniform_like.onnx")


if __name__ == '__main__':
    random_uniform_like()
