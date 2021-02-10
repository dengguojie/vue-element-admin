import onnx
from onnx import helper
from onnx import TensorProto

def make_flatten_input_4d_axis_neg1():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2 * 3 * 4, 5])
    node = helper.make_node(
        'Flatten',
        inputs = ['x'],
        outputs = ['y'],
        axis = -1,
    )
    graph = helper.make_graph(
        nodes = [node],
        name = "test_flatten_input_4d_axis_neg1",
        inputs = [x],
        outputs = [y]
    )

    model = helper.make_model(graph, producer_name = "onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_flatten_input_4d_axis_neg1.onnx")
    onnx.checker.check_model(model)

def make_flatten_input_4d_axis_0():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2 * 3 * 4 * 5])
    node = helper.make_node(
        'Flatten',
        inputs = ['x'],
        outputs = ['y'],
        axis = 0,
    )
    graph = helper.make_graph(
        nodes = [node],
        name = "test_flatten_input_4d_axis_0",
        inputs = [x],
        outputs = [y]
    )

    model = helper.make_model(graph, producer_name = "onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_flatten_input_4d_axis_0.onnx")
    onnx.checker.check_model(model)

def make_flatten_input_4d_axis_2():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2 * 3, 4 * 5])
    node = helper.make_node(
        'Flatten',
        inputs = ['x'],
        outputs = ['y'],
        axis = 2,
    )
    graph = helper.make_graph(
        nodes = [node],
        name = "test_flatten_input_4d_axis_2",
        inputs = [x],
        outputs = [y]
    )

    model = helper.make_model(graph, producer_name = "onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_flatten_input_4d_axis_2.onnx")
    onnx.checker.check_model(model)

def make_flatten_input_4d_axis_3():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2 * 3 * 4, 5])
    node = helper.make_node(
        'Flatten',
        inputs = ['x'],
        outputs = ['y'],
        axis = 3,
    )
    graph = helper.make_graph(
        nodes = [node],
        name = "test_flatten_input_4d_axis_3",
        inputs = [x],
        outputs = [y]
    )

    model = helper.make_model(graph, producer_name = "onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_flatten_input_4d_axis_3.onnx")
    onnx.checker.check_model(model)

def make_flatten_input_4d_axis_4():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2 * 3 * 4 * 5, 1])
    node = helper.make_node(
        'Flatten',
        inputs = ['x'],
        outputs = ['y'],
        axis = 4,
    )
    graph = helper.make_graph(
        nodes = [node],
        name = "test_flatten_input_4d_axis_4",
        inputs = [x],
        outputs = [y]
    )

    model = helper.make_model(graph, producer_name = "onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_flatten_input_4d_axis_4.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_flatten_input_4d_axis_neg1()
    make_flatten_input_4d_axis_0()
    make_flatten_input_4d_axis_2()
    make_flatten_input_4d_axis_3()
    make_flatten_input_4d_axis_4()