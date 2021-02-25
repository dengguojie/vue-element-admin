import onnx
from onnx import helper


def unsqueeze_negative_axes():
    node = helper.make_node('Unsqueeze',
                            inputs=['x'],
                            outputs=['y'],
                            axes=[-2],
                            name='test_unsqueeze_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_unsqueeze_1",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 1, 5])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 1, 1, 5])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_unsqueeze_case_1.onnx")


if __name__ == '__main__':
    unsqueeze_negative_axes()
