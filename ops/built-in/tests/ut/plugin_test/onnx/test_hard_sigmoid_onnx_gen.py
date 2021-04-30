import onnx
from onnx import helper

def make_hard_sigmoid_case_1():
    node = helper.make_node('HardSigmoid',
                            inputs=['X'],
                            outputs=['Y'],
                            alpha=0.1,
                            beta=0.1,
                            name='test_hard_sigmoid_case_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_hard_sigmoid_case_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [12, 4, 3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [12, 4, 3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 1
    onnx.save(model, "./test_hard_sigmoid_case_1.onnx")
    onnx.checker.check_model(model)

def make_hard_sigmoid_case_2():
    node = helper.make_node('HardSigmoid',
                            inputs=['X'],
                            outputs=['Y'],
                            name='test_hard_sigmoid_case_2')
    graph = helper.make_graph(
        nodes=[node],
        name="test_hard_sigmoid_case_2",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [56, 7])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [56, 7])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 6
    onnx.save(model, "./test_hard_sigmoid_case_2.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_hard_sigmoid_case_1()
    make_hard_sigmoid_case_2()
