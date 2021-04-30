import onnx
from onnx import helper


def make_anti():
    node = helper.make_node('AscendAntiQuant',
                            inputs=['A', 'B'],
                            outputs=['C'],
                            scale=1.0,
                            offset=1.0,
                            dtype=1,
                            sqrt_mode=1,
                            name='test_anti_quant_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_anti_quant_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [2, 3, 4]),
                helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT16, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT16, [2, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_anti_quant_1.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_anti()
