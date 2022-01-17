import onnx
from onnx import helper
from onnx import TensorProto

def make_moving_sum_with_sigmoid(version):
    alpha = helper.make_tensor_value_info('alpha', TensorProto.FLOAT, [-1])
    energy = helper.make_tensor_value_info('energy', TensorProto.FLOAT, [-1])
    frame_size = helper.make_tensor_value_info('frame_size', TensorProto.INT32, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [-1])

    node_def = helper.make_node(
        'MovingSumWithSigmoid',
        inputs=['alpha','energy','frame_size'],
        outputs=['y'],
        window_size=10
    )

    graph = helper.make_graph(
        [node_def],
        'test_moving_sum_with_sigmoid',
        inputs=[alpha,energy,frame_size],
        outputs=[y]
    )

    model = helper.make_model(graph, producer_name="onnx_parser_test")
    model.opset_import[0].version = version
    onnx.save(model, "./make_moving_sum_with_sigmoid.onnx")
    onnx.checker.check_model(model)



if __name__ == '__main__':
    make_moving_sum_with_sigmoid(8)
    make_moving_sum_with_sigmoid(9)
    make_moving_sum_with_sigmoid(10)
    make_moving_sum_with_sigmoid(11)
    make_moving_sum_with_sigmoid(12)
    make_moving_sum_with_sigmoid(13)

