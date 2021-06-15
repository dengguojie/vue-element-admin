import onnx
from onnx import helper


def make_shrink():
    node_0 = helper.make_node('Shrink',
                              inputs=['M'],
                              outputs=['Y'],
                              bias=0.0,
                              lambd=0.5,
                              name='test_shrink_1')
    node_1 = helper.make_node('Expand',
                              ['X', 'X1'],
                              ['M'],)

    graph = helper.make_graph(
        nodes=[node_0, node_1],
        name="test_shrink_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 1]),
                helper.make_tensor_value_info("X1", onnx.TensorProto.INT64, [3])],

        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [5])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_shrink_case_v11.onnx")


if __name__ == '__main__':
    make_shrink()



