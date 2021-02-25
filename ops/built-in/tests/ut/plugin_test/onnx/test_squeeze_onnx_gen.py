import onnx
from onnx import helper


def squeeze_default():
    node = helper.make_node('Squeeze',
                            inputs=['x'],
                            outputs=['y'],
                            axes=[0],
                            name='test_Squeeze_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_Squeeze_1",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 4, 5])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [3, 4, 5])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_squeeze_case_1.onnx")


def squeeze_negative_axes():
    node = helper.make_node('Squeeze',
                            inputs=['x'],
                            outputs=['y'],
                            axes=[-2],
                            name='test_Squeeze_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_Squeeze_1",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 1, 5])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 5])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_squeeze_case_2.onnx")


if __name__ == '__main__':
    squeeze_default()
    squeeze_negative_axes()
