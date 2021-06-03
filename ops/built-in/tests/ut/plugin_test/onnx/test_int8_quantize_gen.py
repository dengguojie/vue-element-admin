import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_quantize_case_1():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 224, 224])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT8, [2, 3, 224, 224])
    node_def = helper.make_node('Int8Quantize',
                                inputs=['X'],
                                outputs=['Y'],
                                y_scale=0.003921568859368563,
                                Y_zero_point=0,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_quantize",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx_qauntize_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_quantize_case_1.onnx")


if __name__ == '__main__':
    make_quantize_case_1()
