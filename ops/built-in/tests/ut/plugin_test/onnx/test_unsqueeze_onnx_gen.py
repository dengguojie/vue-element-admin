import onnx
from onnx import helper


def unsqueeze_negative_axes(version_num):
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
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_unsqueeze_case_V{}.onnx".format(version_num))


if __name__ == '__main__':
    version_t = (9, 11, 12)
    for i in version_t:
        unsqueeze_negative_axes(i)
