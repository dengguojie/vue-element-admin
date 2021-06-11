import onnx
from onnx import helper


def make_instance():
    node = helper.make_node('InstanceNormalization',
                            inputs=['X', 'S', 'bias'],
                            outputs=['Y'],
                            epsilon=1e-5,
                            name='test_tan_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_instancenormalization_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 2, 1, 1, 2]),
                helper.make_tensor_value_info("S", onnx.TensorProto.FLOAT, [2]),
                helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [2])],

        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2, 1, 1, 2])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_instancenormalization_case_v11.onnx")


if __name__ == '__main__':
    make_instance()

