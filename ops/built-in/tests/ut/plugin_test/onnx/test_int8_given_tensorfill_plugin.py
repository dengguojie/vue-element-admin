import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_givenint_tensorfill_case_1():
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [2,3])
    Z = helper.make_tensor_value_info("Z", TensorProto.INT8, [2,3])
    node_def = helper.make_node('Int8GivenTensorFill',
                                inputs=[],
                                outputs=['Y'],
                                values="100,101,102,103,104,105",
                                shape=[2,3],
                                zero_point=128,
                                )
    node_def1 = helper.make_node('Add',
                                inputs=['X', 'Y'],
                                outputs=['Z'],
                                )
    graph = helper.make_graph(
        [node_def, node_def1],
        "test_dequantize",
        [X],
        [Z],
    )

    model = helper.make_model(graph, producer_name="onnx-relu_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_given_tensorfill_case_1.onnx")

def make_givenint_tensorfill_case_2():
    X = helper.make_tensor_value_info("X", TensorProto.UINT8, [2,3])
    Z = helper.make_tensor_value_info("Z", TensorProto.UINT8, [2,3])
    node_def = helper.make_node('Int8GivenTensorFill',
                                inputs=[],
                                outputs=['Y'],
                                values="100,101,102,103,104,105",
                                shape=[2,3],
                                zero_point=0,
                                )
    node_def1 = helper.make_node('Add',
                                inputs=['X', 'Y'],
                                outputs=['Z'],
                                )
    graph = helper.make_graph(
        [node_def, node_def1],
        "test_dequantize",
        [X],
        [Z],
    )

    model = helper.make_model(graph, producer_name="onnx-relu_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_given_tensorfill_case_2.onnx")

if __name__ == '__main__':
    make_givenint_tensorfill_case_1()
    make_givenint_tensorfill_case_2()
