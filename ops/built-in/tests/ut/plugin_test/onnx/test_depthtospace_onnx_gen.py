import onnx
from onnx import helper


def make_depthtospace_1():
    node = helper.make_node('DepthToSpace',
                            inputs=['x'],
                            outputs=['y'],
                            blocksize=2,
                            mode="DCR",
                            name='test_depthtospace_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_depthtospace_1",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 8, 2, 3])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 4, 6])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_depthtospace_case_1.onnx")


if __name__ == '__main__':
    make_depthtospace_1()
