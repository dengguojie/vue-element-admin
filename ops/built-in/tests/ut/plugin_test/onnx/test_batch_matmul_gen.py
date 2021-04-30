import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_batch_matmul():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
    X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [3, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [3,3])
    node_def = helper.make_node('BatchMatMul',
                                inputs=['X', 'X1'],
                                outputs=['Y'],
                                )
    graph = helper.make_graph(
        [node_def],
        "test_cast",
        [X, X1],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-cast_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_batch_matmul.onnx")


if __name__ == '__main__':
   make_batch_matmul()
