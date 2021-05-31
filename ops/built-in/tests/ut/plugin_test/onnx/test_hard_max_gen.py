import onnx
from onnx import helper

def make_hardmax_case_1():
    node = helper.make_node('Hardmax',
                            inputs=['X'],
                            outputs=['Y'],
                            axis=1,
                            name='test_hard_max_case_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_hard_max_case_1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [12, 4, 3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [12, 4, 3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_hard_max_case_1.onnx")
    onnx.checker.check_model(model)

if __name__ == "__main__":
    make_hardmax_case_1()