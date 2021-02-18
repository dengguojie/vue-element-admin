import onnx
from onnx import helper


def make_argmax_1():
    node = helper.make_node('ArgMax',
                            inputs=['data'],
                            outputs=['result'],
                            keepdims=1,
                            name='test_argmax_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_argmaxx_1",
        inputs=[helper.make_tensor_value_info(
            "data", onnx.TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info(
            "result", onnx.TensorProto.FLOAT, [1, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_argmax_case_1.onnx")


if __name__ == '__main__':
    make_argmax_1()
