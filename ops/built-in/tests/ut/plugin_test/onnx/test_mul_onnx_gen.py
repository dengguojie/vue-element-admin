import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_mul(version_number):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 5])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4, 5])
    node_def = helper.make_node('Mul',
                                inputs=['X','Y'],
                                outputs=['Z'],
                                )
    graph = helper.make_graph(
        [node_def],
        "test_mul",
        [X,Y],
        [Z],
    )

    model = helper.make_model(graph, producer_name="onnx-mul_test")
    model.opset_import[0].version = version_number
    onnx.save(model, "./test_mul_case_version_{}.onnx".format(version_number))


if __name__ == '__main__':
    mul_list = (9,11,12,13)
    for i in mul_list:
        make_mul(i)
