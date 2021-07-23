import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def random_normal():
    shape = [3, 3]
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

    # Create two node
    random = helper.make_node(
        'RandomNormal',
        [],
        ['y'],
        dtype=1,
        mean=0.5,
        scale=1.5,
        seed=0.1,
        shape=shape,
    )

    add = helper.make_node(
        'Add',  # node name
        ['x', 'y'],  # inputs
        ['output'],  # outputs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [random, add],
        'randomnormal',
        inputs=[x],
        outputs=[output],
    )

    model_def = helper.make_model(graph_def, producer_name='randomNormal-onnx')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_random_normal_case_1.onnx")


if __name__ == '__main__':
    random_normal()
