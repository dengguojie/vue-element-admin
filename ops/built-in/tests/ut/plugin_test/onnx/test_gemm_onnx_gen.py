import onnx
from onnx import helper
from onnx import TensorProto


def make_gemm_v11():
    """v11"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [4, 3])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [5, 4])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [1, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])
    node_def = helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        alpha=0.25,
        beta=0.35,
        transA=1,
        transB=1
    )

    graph = helper.make_graph(
        [node_def],
        'Gemm_v11',
        [a, b, c],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_gemm_case_v11.onnx")
    onnx.checker.check_model(model)


def make_gemm_v12():
    """v12"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [4, 3])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [5, 4])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [1, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])
    node_def = helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        alpha=0.25,
        beta=0.35,
        transA=1,
        transB=1
    )

    graph = helper.make_graph(
        [node_def],
        'Gemm_v12',
        [a, b, c],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_gemm_case_v12.onnx")
    onnx.checker.check_model(model)


def make_gemm_v13():
    """v13"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [4, 3])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [5, 4])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [1, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])
    node_def = helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        alpha=0.25,
        beta=0.35,
        transA=1,
        transB=1
    )

    graph = helper.make_graph(
        [node_def],
        'Gemm_v13',
        [a, b, c],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_gemm_case_v13.onnx")
    onnx.checker.check_model(model)


def make_gemm_v9():
    """v9"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [4, 3])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [5, 4])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [1, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])
    node_def = helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        alpha=0.25,
        beta=0.35,
        transA=1,
        transB=1
    )

    graph = helper.make_graph(
        [node_def],
        'Gemm_v9',
        [a, b, c],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_gemm_case_v9.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_gemm_v9()
    make_gemm_v11()
    make_gemm_v12()
    make_gemm_v13()
