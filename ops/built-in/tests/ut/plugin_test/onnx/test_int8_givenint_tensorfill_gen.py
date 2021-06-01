import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_givenint_tensorfill_case_1():
    X = helper.make_tensor_value_info("X", TensorProto.INT32, [5])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [5])
    Z = helper.make_tensor_value_info("Z", TensorProto.INT32, [5])
    node_def = helper.make_node('Int8GivenIntTensorFill',
                                inputs=[],
                                outputs=['Y'],
                                values=[0,1,2,3,4],
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
    onnx.save(model, "./test_int8_givenint_tensorfill_case_1.onnx")

if __name__ == '__main__':
    make_givenint_tensorfill_case_1()
