import onnx
from onnx import helper
from onnx import TensorProto


def make_matmul_v11():
    """v11"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [2, 4, 3])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [2, 3, 3])
    node_def = helper.make_node(
        'MatMul',
        inputs=['a', 'b'],
        outputs=['c']
    )

    graph = helper.make_graph(
        [node_def],
        'MatMul_v11',
        [a, b],
        [c],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_matmul_case_v11.onnx")
    onnx.checker.check_model(model)

def make_matmul_transa_v11():
    """v11"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3, 3])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [2, 3, 3])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [2, 3, 3])
    node_def = helper.make_node(
        'MatMul',
        inputs=['a', 'b'],
        outputs=['c'],
        transA=1
    )

    graph = helper.make_graph(
        [node_def],
        'MatMul_v11',
        [a, b],
        [c],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_matmul_case_transa_v11.onnx")

def make_matmul_transb_v11():
    """v11"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3, 3])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [2, 3, 3])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [2, 3, 3])
    node_def = helper.make_node(
        'MatMul',
        inputs=['a', 'b'],
        outputs=['c'],
        transB=1
    )

    graph = helper.make_graph(
        [node_def],
        'MatMul_v11',
        [a, b],
        [c],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_matmul_case_transb_v11.onnx")


def make_matmul_v12():
    """v12"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [2, 4, 3])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [2, 3, 3])
    node_def = helper.make_node(
        'MatMul',
        inputs=['a', 'b'],
        outputs=['c']
    )

    graph = helper.make_graph(
        [node_def],
        'MatMul_v12',
        [a, b],
        [c],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_matmul_case_v12.onnx")
    onnx.checker.check_model(model)


def make_matmul_v13():
    """v13"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [2, 4, 3])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [2, 3, 3])
    node_def = helper.make_node(
        'MatMul',
        inputs=['a', 'b'],
        outputs=['c']
    )

    graph = helper.make_graph(
        [node_def],
        'MatMul_v13',
        [a, b],
        [c],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_matmul_case_v13.onnx")
    onnx.checker.check_model(model)


def make_matmul_v9():
    """v9"""
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [2, 4, 3])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [2, 3, 3])
    node_def = helper.make_node(
        'MatMul',
        inputs=['a', 'b'],
        outputs=['c']
    )

    graph = helper.make_graph(
        [node_def],
        'MatMul_v9',
        [a, b],
        [c],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_matmul_case_v9.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_matmul_v9()
    make_matmul_v11()
    make_matmul_v12()
    make_matmul_v13()
    make_matmul_transa_v11()
    make_matmul_transb_v11()
