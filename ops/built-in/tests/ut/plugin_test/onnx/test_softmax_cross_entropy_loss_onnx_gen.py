import onnx
from onnx import helper
from onnx import TensorProto
 
def softmax_cross_entropy_loss_v12():
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [3, 5, 6])
    labels = helper.make_tensor_value_info('labels', TensorProto.INT32, [3, 6])
    weights = helper.make_tensor_value_info('weights', TensorProto.FLOAT, [5,])
    loss = helper.make_tensor_value_info('loss', TensorProto.FLOAT, [3, 6])
    log_prop = helper.make_tensor_value_info('log_prop', TensorProto.FLOAT, [3, 5, 6])

    node_def = helper.make_node(
        'SoftmaxCrossEntropyLoss',
        inputs=['scores','labels','weights'],
        outputs=['loss','log_prop'],
        ignore_index = 0,
        reduction='none'
    )

    graph = helper.make_graph(
        [node_def],
        'softmax_cross_entropy_loss',
        inputs=[scores,labels,weights],
        outputs=[loss,log_prop],
    )
    model = helper.make_model(graph, producer_name="onnx_parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_SoftmaxCrossEntropyLoss_case_v12.onnx")
    onnx.checker.check_model(model)

def softmax_cross_entropy_loss_v13():
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [3, 5, 6])
    labels = helper.make_tensor_value_info('labels', TensorProto.INT32, [3, 6])
    weights = helper.make_tensor_value_info('weights', TensorProto.FLOAT, [5,])
    loss = helper.make_tensor_value_info('loss', TensorProto.FLOAT, [3, 6])
    log_prop = helper.make_tensor_value_info('log_prop', TensorProto.FLOAT, [3, 5, 6])

    node_def = helper.make_node(
        'SoftmaxCrossEntropyLoss',
        inputs=['scores','labels','weights'],
        outputs=['loss','log_prop'],
        ignore_index = 0,
        reduction='none'
    )

    graph = helper.make_graph(
        [node_def],
        'softmax_cross_entropy_loss',
        inputs=[scores,labels,weights],
        outputs=[loss,log_prop],
    )
    model = helper.make_model(graph, producer_name="onnx_parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_SoftmaxCrossEntropyLoss_case_v13.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    softmax_cross_entropy_loss_v12()
    softmax_cross_entropy_loss_v13()