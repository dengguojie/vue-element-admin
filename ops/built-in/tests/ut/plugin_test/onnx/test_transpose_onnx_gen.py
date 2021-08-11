import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_transpose_case_1():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3, 2])
    node_def = helper.make_node('Transpose',
                                inputs=['X'],
                                outputs=['Y'],
                                perm=[2, 1, 0]
                                )
    graph = helper.make_graph(
        [node_def],
        "test_transpose_case_1",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-transpose_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_transpose_case_1.onnx")


def make_transpose_case_2():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 1, 3])
    node_def = helper.make_node('Transpose',
                                inputs=['X'],
                                outputs=['Y'],
                                perm=[1, 0, 2],
                                )
    graph = helper.make_graph(
        [node_def],
        "test_transpose_case_2",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-transpose_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_transpose_case_2.onnx")


if __name__ == '__main__':
    make_transpose_case_1()
    make_transpose_case_2()
