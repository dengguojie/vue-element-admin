import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_cast(version_num):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [3,3])
    node_def = helper.make_node('Cast',
                                inputs=['X'],
                                outputs=['Y'],
                                to =getattr(TensorProto,"FLOAT16")
                                )
    graph = helper.make_graph(
        [node_def],
        "test_cast",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-cast_test")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_cast_version_{}.onnx".format(version_num))


if __name__ == '__main__':
    cast_list = (9,11,12,13)
    for i in cast_list:
        make_cast(i)
