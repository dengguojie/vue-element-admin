import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_leaky_relu_case_1():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    node_def = helper.make_node('LeakyRelu',
                                inputs=['X'],
                                outputs=['Y'],
                                alpha=0.1
                                )
    graph = helper.make_graph(
        [node_def],
        "test_LeakyRelu",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-relu_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_leaky_relu_case_1.onnx")

def make_leaky_relu_case_default():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3,4,5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3,4,5])
    node_def = helper.make_node('LeakyRelu',
                                inputs=['X'],
                                outputs=['Y'],
                                )
    graph = helper.make_graph(
        [node_def],
        "test_LeakyRelu",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-relu_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_leaky_relu_case_default.onnx")
if __name__ == '__main__':
    make_leaky_relu_case_1()
    make_leaky_relu_case_default()
