import onnx
from onnx import helper


def make_trilu():
    node = helper.make_node('Trilu',
                            inputs=['X'],
                            outputs=['Y'],
                            upper=1,
                            name='test_trilu_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_trilu_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.INT64, [4,5])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.INT64, [4,5])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 14
    onnx.save(model, "./test_trilu_case_v14.onnx")

def make_trilu1():
    node = helper.make_node('Trilu',
                            inputs=['X'],
                            outputs=['Y'],
                            upper=0,
                            name='test_trilu_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_trilu_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.INT64, [4,5])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.INT64, [4,5])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 14
    onnx.save(model, "./test_trilu_case_v14_1.onnx")

if __name__ == '__main__':
    make_trilu()
    make_trilu1()