import onnx
from onnx import helper


def make_xor_1():
    node = helper.make_node('Xor',
                            inputs=['A', 'B'],
                            outputs=['C'])
    graph = helper.make_graph(
        nodes=[node],
        name="test_xor",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [3, 4, 5, 6]),
                helper.make_tensor_value_info("B", onnx.TensorProto.BOOL, [5, 6])],
        outputs=[helper.make_tensor_value_info(
            "C", onnx.TensorProto.FLOAT, [3, 4, 5, 6])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_xor_case_1.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_xor_1()
