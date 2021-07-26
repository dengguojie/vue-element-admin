import onnx
from onnx import helper


def make_multinomial():
    node = helper.make_node('Multinomial',
                            inputs=['A'],
                            outputs=['C'],
                            dtype=7,
                            sample_size=2,
                            seed=4.0,
                            name='test_add_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_add_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [2, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT16, [2, 2])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_multinomal_case_1.onnx")
    onnx.checker.check_model(model)

def make_multinomial_fail():
    node = helper.make_node('Multinomial',
                            inputs=['A'],
                            outputs=['C'],
                            dtype=71,
                            sample_size=2,
                            seed=4.0,
                            name='test_add_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_add_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [2, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT16, [2, 2])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_multinomal_case_2.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_multinomial()
    make_multinomial_fail()
