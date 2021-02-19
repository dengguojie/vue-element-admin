import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_reducemean_1():
    X = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 2, 2])
    Y = helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [1, 1, 1])
    node_def = helper.make_node('ReduceMean',
                                inputs=['data'],
                                outputs=['reduced'],
                                keepdims=1,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_reducemean_case_1",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-reducemean_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_reducemean_case_1.onnx")


def make_reducemean_2():
    X = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 3, 2])
    Y = helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [4, 2])
    node_def = helper.make_node('ReduceMean',
                                inputs=['data'],
                                outputs=['reduced'],
                                keepdims=0,
                                axes=[1]
                                )
    graph = helper.make_graph(
        [node_def],
        "test_reducemean_case_2",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-reducemean_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_reducemean_case_2.onnx")


if __name__ == '__main__':
    make_reducemean_1()
    make_reducemean_2()
