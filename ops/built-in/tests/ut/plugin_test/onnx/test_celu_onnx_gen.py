import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_celu(version_number):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 5])
    node_def = helper.make_node('Celu',
                                inputs=['X'],
                                outputs=['Y'],
                                )
    graph = helper.make_graph(
        [node_def],
        "test_celu",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-celu_test")
    model.opset_import[0].version = version_number
    onnx.save(model, "./test_relu_case_version_{}.onnx".format(version_number))

if __name__ == '__main__':
    celu_list = (12)
    for i in celu_list:
        make_celu(i)
