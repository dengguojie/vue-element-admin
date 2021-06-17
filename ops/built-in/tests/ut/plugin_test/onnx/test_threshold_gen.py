import onnx
from onnx import helper


def make_threshold_v11():
    node = helper.make_node('ThresholdedRelu',
                            inputs=['X'],
                            outputs=['Y'],
                            alpha=1.0,
                            name='test_threshold_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_threshold_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_threshold_case_v11.onnx")

def make_threshold_v10():
    node = helper.make_node('ThresholdedRelu',
                            inputs=['X'],
                            outputs=['Y'],
                            alpha=1.0,
                            name='test_threshold_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_threshold_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 10
    onnx.save(model, "./test_threshold_case_v10.onnx")

if __name__ == '__main__':
    make_threshold_v10()
    make_threshold_v11()