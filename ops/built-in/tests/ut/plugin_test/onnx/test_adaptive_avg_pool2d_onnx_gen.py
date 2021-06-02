import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_adaptive_avg_pool2d_case_1():
    X = helper.make_tensor_value_info("X", TensorProto.INT8, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT8, [1, 3, 40, 40])
    node_def = helper.make_node('AdaptiveAvgPool2d',
                                inputs=['X'],
                                outputs=['Y'],
                                output_size=[40, 40]
                                )
    graph = helper.make_graph(
        [node_def],
        "test_adaptive_avg_pool2d",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx_adaptive_avg_pool2d")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_adaptive_avg_pool2d_case_1.onnx")


if __name__ == '__main__':
    make_adaptive_avg_pool2d_case_1()
