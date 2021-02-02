import onnx
from onnx import helper

def make_constantofshape_float_ones():
    tensor_value = helper.make_tensor("value", onnx.TensorProto.FLOAT,
                                       [1], [1])
    node = helper.make_node(
        'ConstantOfShape',
        inputs=['x'],
        outputs=['y'],
        value=tensor_value
    )
    graph = helper.make_graph(
        nodes=[node],
        name="test_constantofshape_float_ones",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.INT64, [3])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4, 3, 2])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_constantofshape_float_ones.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_constantofshape_float_ones()