import onnx
from onnx import helper
from onnx import TensorProto

def make_lrn():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 20, 10, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 20, 10, 5])
    node_def = helper.make_node(
        'LRN',
        inputs=['x'],
        outputs=['y'],
        alpha=0.0001,
        beta=0.75,
        bias=1.0,
        size=9
    )

    graph = helper.make_graph(
        [node_def],
        'test_lrn',
        inputs=[x],
        outputs=[y]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_lrn_case_1.pb")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_lrn()
