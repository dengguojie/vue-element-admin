import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def GroupNormRelu():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3,4,5,6])
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3,4,5,6])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3,4,5,6])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3,4,5,6])

    # Create two node
    random = helper.make_node(
        'GroupNormRelu',
        ['x', 'x1', 'x2'],
        ['output'],
        num_groups=2,
        eps=0.5,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [random],
        'randomnormal',
        inputs=[x,x1,x2],
        outputs=[output],
    )

    model_def = helper.make_model(graph_def, producer_name='randomNormal-onnx')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_groupnorm_relu_case_1.onnx")


if __name__ == '__main__':
    GroupNormRelu()
