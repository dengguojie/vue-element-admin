import onnx
from onnx import helper
from onnx import TensorProto

def make_slice():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [20, 10, 5])
    start = helper.make_tensor('start', TensorProto.INT64, [1,3], [0,0,3])
    end = helper.make_tensor('end', TensorProto.INT64, [1,3], [20,10,4])
    axis = helper.make_tensor('axis', TensorProto.INT64, [1,3], [0,1,2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [20,10,1])
    node_def=helper.make_node(
        'Slice',
        ['x','start','end','axis'],
        ['y']
    )

    graph=helper.make_graph(
        [node_def],
        'test_slice',
        [x],
        [y],
        [start, end, axis],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_slice_case_1.pb")
    onnx.checker.check_model(model)

def make_slice1():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [20, 10, 5])
    start = helper.make_tensor('start', TensorProto.INT64, [1,3], [0,0,3])
    end = helper.make_tensor('end', TensorProto.INT64, [1,3], [20,10,4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [20,10,1])
    node_def=helper.make_node(
        'Slice',
        ['x','start','end'],
        ['y']
    )

    graph=helper.make_graph(
        [node_def],
        'test_slice',
        [x],
        [y],
        [start, end],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_slice_case_2.pb")
    onnx.checker.check_model(model)

def make_slice2():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [20, 10, 5])
    start = helper.make_tensor('start', TensorProto.INT64, [1,3], [0,0,3])
    end = helper.make_tensor('end', TensorProto.INT64, [1,3], [20,10,4])
    axis = helper.make_tensor('axis', TensorProto.INT64, [1,3], [0,1,2])
    steps = helper.make_tensor('step', TensorProto.INT64, [1,3], [1,1,1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [20,10,1])
    node_def=helper.make_node(
        'Slice',
        ['x','start','end','axis', 'step'],
        ['y']
    )

    graph=helper.make_graph(
        [node_def],
        'test_slice',
        [x],
        [y],
        [start, end, axis, steps],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_slice_case_3.pb")
    onnx.checker.check_model(model)

def make_slice3():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [20, 10, 5])
    start = helper.make_tensor('start', TensorProto.INT64, [1,3], [0,0,3])
    end = helper.make_tensor('end', TensorProto.INT64, [1,3], [20,10,4])
    axis = helper.make_tensor('axis', TensorProto.INT64, [1,3], [0,1,2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [20,10,1])
    node_def=helper.make_node(
        'Slice',
        ['x','start','end','axis'],
        ['y']
    )

    graph=helper.make_graph(
        [node_def],
        'test_slice',
        [x],
        [y],
        [start, end, axis],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_slice_case_4.pb")
    onnx.checker.check_model(model)

def make_slice4():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [20, 10, 5])
    start = helper.make_tensor('start', TensorProto.INT64, [1,3], [0,0,3])
    end = helper.make_tensor('end', TensorProto.INT64, [1,3], [20,10,4])
    axis = helper.make_tensor('axis', TensorProto.INT64, [1,3], [0,1,2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [20,10,1])
    node_def=helper.make_node(
        'Slice',
        ['x','start','end','axis'],
        ['y']
    )

    graph=helper.make_graph(
        [node_def],
        'test_slice',
        [x],
        [y],
        [start, end, axis],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_slice_case_5.pb")
    onnx.checker.check_model(model)

def make_slice5():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [20, 10, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [20,10,1])
    node_def=helper.make_node(
        'Slice',
        ['x'],
        ['y'],
        axes=[0,1],
        starts=[0,0,0],
        ends=[3,10,5]
    )

    graph=helper.make_graph(
        [node_def],
        'test_slice',
        [x],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_slice_case_6.pb")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_slice()
    make_slice1()
    make_slice2()
    make_slice3()
    make_slice4()
    make_slice5()
