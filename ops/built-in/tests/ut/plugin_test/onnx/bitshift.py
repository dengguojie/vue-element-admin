import onnx
from onnx import helper
from onnx import TensorProto

def bit_lift(i):
    x = helper.make_tensor_value_info('x', TensorProto.UINT16, [3])
    y = helper.make_tensor_value_info('y', TensorProto.UINT16, [3])
    z = helper.make_tensor_value_info('z', TensorProto.UINT16, [3])

    node = helper.make_node('BitShift',
                            inputs=['x', 'y'],
                            outputs=['z'],
                            direction="LEFT",)


    graph = helper.make_graph(
        [node],
        'BitShift',
        [x, y],
        [z],
    )

    model = helper.make_model(graph, producer_name="onnx-bitshift_test_3")
    model.opset_import[0].version = i
    onnx.save(model, "./test_bitshift_left_v{}.onnx".format(i))

def bit_right(i):
    x = helper.make_tensor_value_info('x', TensorProto.INT8, [3])
    y = helper.make_tensor_value_info('y', TensorProto.INT8, [3])
    z = helper.make_tensor_value_info('z', TensorProto.INT8, [3])

    node = helper.make_node('BitShift',
                            inputs=['x', 'y'],
                            outputs=['z'],
                            direction="RIGHT",)


    graph = helper.make_graph(
        [node],
        'BitShift',
        [x, y],
        [z],
    )

    model = helper.make_model(graph, producer_name="onnx-bitshift_test_3")
    model.opset_import[0].version = i
    onnx.save(model, "./test_bitshift_right_v{}.onnx".format(i))

def bit_other(i):
    x = helper.make_tensor_value_info('x', TensorProto.UINT16, [3])
    y = helper.make_tensor_value_info('y', TensorProto.UINT16, [3])
    z = helper.make_tensor_value_info('z', TensorProto.UINT16, [3])

    node = helper.make_node('BitShift',
                            inputs=['x', 'y'],
                            outputs=['z'],
                            direction="RILWD",)


    graph = helper.make_graph(
        [node],
        'BitShift',
        [x, y],
        [z],
    )

    model = helper.make_model(graph, producer_name="onnx-bitshift_test_3")
    model.opset_import[0].version = i
    onnx.save(model, "./test_bitshift_other_v{}.onnx".format(i))

if __name__ == '__main__':
    version_t = (11, 12, 13)
    for i in version_t:
        bit_lift(i)
        bit_right(i)
        bit_other(i)
