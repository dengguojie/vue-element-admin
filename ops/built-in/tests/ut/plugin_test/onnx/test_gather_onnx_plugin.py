import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def make_gather(version_num):
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [5, 4, 3, 2])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5, 3, 3, 2])
    
    node_def = helper.make_node('Gather',
                                inputs=['data','indices'],
                                outputs=['Y'],
                                axis=1,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_gather",
        [data,indices],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-gather_test")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_gather_case_V{}.onnx".format(version_num))


if __name__ == '__main__':
    gather_list = (9, 11, 12, 13)
    for i in gather_list:
        make_gather(i)
