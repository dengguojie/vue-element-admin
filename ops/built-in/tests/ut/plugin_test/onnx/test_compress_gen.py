import onnx
from onnx import helper


def make_compress_1():
    node = helper.make_node('Compress',
                            inputs=['A', 'B'],
                            outputs=['C'],
                            axis=-1)
    graph = helper.make_graph(
        nodes=[node],
        name="test_compress_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [3, 2]),
                helper.make_tensor_value_info("B", onnx.TensorProto.BOOL, [2, ])],
        outputs=[helper.make_tensor_value_info(
            "C", onnx.TensorProto.FLOAT, [3, 1])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_compress_case_1.onnx")
    onnx.checker.check_model(model)


def make_compress_2():
    node = helper.make_node('Compress',
                            inputs=['A', 'B'],
                            outputs=['C'])
    graph = helper.make_graph(
        nodes=[node],
        name="test_compress_2",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [3, 2]),
                helper.make_tensor_value_info("B", onnx.TensorProto.BOOL, [5, ])],
        outputs=[helper.make_tensor_value_info(
            "C", onnx.TensorProto.FLOAT, [2, ])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_compress_case_2.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_compress_1()
    make_compress_2()
