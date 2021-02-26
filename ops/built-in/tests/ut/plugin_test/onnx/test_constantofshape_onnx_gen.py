import onnx
from onnx import helper

def make_constantofshape_float_ones(version_num):
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
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_constantofshape_V{}.onnx".format(version_num))
    onnx.checker.check_model(model)

if __name__ == '__main__':
    version_t = (9, 11, 12, 13)
    for i in version_t:
        make_constantofshape_float_ones(i)