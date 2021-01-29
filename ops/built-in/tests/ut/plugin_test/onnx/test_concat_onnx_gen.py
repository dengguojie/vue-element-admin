import onnx
from onnx import helper


def make_concat():
    node = helper.make_node('Concat',
                            inputs=['A'],
                            outputs=['C'],
                            axis=0,
                            name='test_add_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_add_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [2, 3, 4])]

    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_concat_case_1.pb")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_concat()
