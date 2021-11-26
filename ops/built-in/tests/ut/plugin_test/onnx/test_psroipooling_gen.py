import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def psroi():
    shape = [2, 16*20*20,20,20]
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [2,5,16])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

    # Create two node
    random = helper.make_node(
        'PSROIPooling',
        ['x', 'x1'],
        ['output'],
        spatial_scale=0.0625,
        output_dim=16,
        group_size=20,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [random],
        'randomnormal',
        inputs=[x,x1],
        outputs=[output],
    )

    model_def = helper.make_model(graph_def, producer_name='randomNormal-onnx')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_psroipooling_case_1.onnx")


if __name__ == '__main__':
    psroi()
