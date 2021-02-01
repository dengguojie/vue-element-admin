import onnx
from onnx import helper


def make_average_pool_1():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[2, 2],
                            name='test_average_pool_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_1",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 31, 31])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_average_pool_case_1.onnx")


def make_average_pool_2():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[5, 5],
                            strides=[3, 3],
                            name='test_average_pool_2')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_2",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 10, 10])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_2")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_average_pool_case_2.onnx")


if __name__ == '__main__':
    make_average_pool_1()
    make_average_pool_2()
