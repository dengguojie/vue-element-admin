import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_transpose_case_1():
    axes = [0, 2, 3, 1]
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 2, 3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT8, [1, 3, 4, 2])
    node_def = helper.make_node('Int8Transpose',
                                inputs=['X'],
                                outputs=['Y'],
                                axes=axes,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_transpose",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="zyw_onnx_transpose")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_transpose.onnx")

if __name__ == '__main__':
    make_transpose_case_1()
