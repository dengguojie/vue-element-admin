import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_clip_V9():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 5])
    node_def = helper.make_node('Clip',
                                inputs=['X'],
                                outputs=['Y'],
                                max = 1,
                                min = -1,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_clip_case_V9",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-mul_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_clip_case_V9.onnx")


def make_clip(version_num):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])
    max = helper.make_tensor_value_info("max", TensorProto.FLOAT, [])
    min = helper.make_tensor_value_info("min", TensorProto.FLOAT, [])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 5])
    
    node_def = helper.make_node('Clip',
                                inputs=['X','max','min'],
                                outputs=['Y'],
                                )
    graph = helper.make_graph(
        [node_def],
        "test_clip",
        [X,max,min],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-clip_test")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_clip_case_V{}.onnx".format(version_num))


if __name__ == '__main__':
    make_clip_V9()
    clip_list = (11, 12, 13)
    for i in clip_list:
        make_clip(i)
