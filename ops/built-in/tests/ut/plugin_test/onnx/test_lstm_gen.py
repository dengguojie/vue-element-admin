import onnx
from onnx import helper

def make_lstm_case_1():
    node = helper.make_node('LSTM',
                            inputs=['X','X1', 'X2'],
                            outputs=['Y', 'Y1'],
                            hidden_size=5,
                            direction="forward",
                            name='test_lstm_case_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_lstm_case_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [12, 4, 3]),
                helper.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, [12, 4, 3]),
                helper.make_tensor_value_info("X2", onnx.TensorProto.FLOAT, [12, 4, 3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [12, 4, 3]),
                 helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [12, 4, 3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_lstm_case_1.onnx")
    onnx.checker.check_model(model)

def make_lstm_case_2():
    node = helper.make_node('LSTM',
                            inputs=['X','X1', 'X2'],
                            outputs=['Y', 'Y1'],
                            direction="forward",
                            name='test_lstm_case_2')
    graph = helper.make_graph(
        nodes=[node],
        name="test_lstm_case_2",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [12, 4, 3]),
                helper.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, [12, 4, 3]),
                helper.make_tensor_value_info("X2", onnx.TensorProto.FLOAT, [12, 4, 3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [12, 4, 3]),
                 helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [12, 4, 3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_lstm_case_2.onnx")
    onnx.checker.check_model(model)


if __name__ == "__main__":
    make_lstm_case_2()
    make_lstm_case_1()