import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_dequantize_case_1():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    node_def = helper.make_node('Int8Dequantize',
                                inputs=['X'],
                                outputs=['Y'],
                                y_scale=0.1789,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_dequantize",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-relu_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_dequantize_case_1.onnx")

if __name__ == '__main__':
    make_dequantize_case_1()
