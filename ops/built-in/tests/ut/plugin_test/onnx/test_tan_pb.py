import onnx
from onnx import helper


def make_tan():
    node = helper.make_node('Tan',
                            inputs=['X'],
                            outputs=['Y'],
                            name='test_tan_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_tan_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_tan_case_v11.onnx")

if __name__ == '__main__':
    make_tan()