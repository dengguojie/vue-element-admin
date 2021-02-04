import onnx
from onnx import helper


def make_min():
    node = helper.make_node('Min',
                            inputs=['x', 'x1'],
                            outputs=['y'],
                            name='test_min_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_min_1",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("x1", onnx.TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_min_case_1.onnx")
    onnx.checker.check_model(model)

def make_min1():
    node = helper.make_node('Min',
                            inputs=['x', 'x1', 'x2'],
                            outputs=['y'],
                            name='test_min_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_min_1",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("x1", onnx.TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("x2", onnx.TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_min_case_2.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_min()
    make_min1()
