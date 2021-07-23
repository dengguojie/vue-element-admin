import onnx
from onnx import helper
from onnx import TensorProto


def make_einsum_v12():
    """v12"""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, 2, 3])
    w = helper.make_tensor_value_info('W', TensorProto.FLOAT, [5, 3, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 2, 4])
    node_def = helper.make_node(
        'Einsum',
        inputs=['x', 'W'],
        outputs=['y'],
        equation="bij,bjk->bik"
    )

    graph = helper.make_graph(
        [node_def],
        'einsum',
        [x, w],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_einsum_case_1.onnx")


if __name__ == '__main__':
    make_einsum_v12()
