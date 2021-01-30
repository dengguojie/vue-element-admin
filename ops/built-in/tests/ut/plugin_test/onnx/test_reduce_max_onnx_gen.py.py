import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def make_reduce_max_1():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3,2,2])
    #keepdims = helper.make_tensor_value_info("keepdims", AttributeProto.INT, [])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1,1,1])
    node_def = helper.make_node('ReduceMax',
                            inputs = ['X'],
                            outputs = ['Y'],
                            keepdims = 1,axes = [1],
    )
    graph = helper.make_graph(
        [node_def],
        "test_reduce_max_case_1",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-reduce_max_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_reduce_max_case_1.onnx")


if __name__ == '__main__':
    make_reduce_max_1()
