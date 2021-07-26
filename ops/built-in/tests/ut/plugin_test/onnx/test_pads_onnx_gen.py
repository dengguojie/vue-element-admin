import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def case_one():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['X'], # inputs
    outputs=['Y'], # outputs
    pads = [0,2,0,0],
    value = 0,
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyx')
    model_def.opset_import[0].version = 9
    onnx.save(model_def, "./test_pads_V9_case.onnx")


def pad_9_mode_reflect():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['X'], # inputs
    outputs=['Y'], # outputs
    pads = [0,2,0,0],
    value = 0,
    mode = "reflect"
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyx')
    model_def.opset_import[0].version = 9
    onnx.save(model_def, "./test_pads_V9_case_mode_reflect.onnx")

def pad_9_mode_fail():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['X'], # inputs
    outputs=['Y'], # outputs
    pads = [0,2,0],
    value = 0,
    mode = "constant"
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyx')
    model_def.opset_import[0].version = 9
    onnx.save(model_def, "./test_pads_V9_case_mode_fail.onnx")

def pad_9_mode_nopad():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['X'], # inputs
    outputs=['Y'], # outputs
    value = 0,
    mode = "constant"
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyx')
    model_def.opset_import[0].version = 9
    onnx.save(model_def, "./test_pads_V9_case_mode_nopad.onnx")

def pad_11_int32():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    pads = helper.make_tensor('paddings', TensorProto.INT32, [4], [0,2,0,0])
    value = helper.make_tensor('constant_values', TensorProto.INT32, [1],[0])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['x','paddings','constant_values'], # inputs
    outputs=['y'], # outputs
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    [pads,value],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyw-onnx-padV3')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_pads_V11_case_INT32.onnx")

def pad_11_float():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    pads = helper.make_tensor('paddings', TensorProto.INT32, [4], [0,2,0,0])
    value = helper.make_tensor('constant_values', TensorProto.FLOAT, [1],[0])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['x','paddings','constant_values'], # inputs
    outputs=['y'], # outputs
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    [pads,value],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyw-onnx-padV3')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_pads_V11_case_float.onnx")

def pad_11_double():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    pads = helper.make_tensor('paddings', TensorProto.INT32, [4], [0,2,0,0])
    value = helper.make_tensor('constant_values', TensorProto.DOUBLE, [1],[0])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['x','paddings','constant_values'], # inputs
    outputs=['y'], # outputs
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    [pads,value],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyw-onnx-padV3')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_pads_V11_case_double.onnx")

def pad_11_int64():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    pads = helper.make_tensor('paddings', TensorProto.INT32, [4], [0,2,0,0])
    value = helper.make_tensor('constant_values', TensorProto.INT64, [1],[0])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['x','paddings','constant_values'], # inputs
    outputs=['y'], # outputs
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    [pads,value],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyw-onnx-padV3')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_pads_V11_case_INT64.onnx")


def pad_11_mode_edge():
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 2])
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4])
    pads = helper.make_tensor('paddings', TensorProto.INT32, [4], [0,2,0,0])
    value = helper.make_tensor('constant_values', TensorProto.INT64, [1],[0])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
    'Pad', # node name
    inputs=['x','paddings','constant_values'], # inputs
    outputs=['y'], # outputs
    mode="edge"
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
    [pads,value],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyw-onnx-padV3')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_pads_V11_case_mode_edge.onnx")


if __name__ == "__main__":
    case_one()
    pad_9_mode_reflect()
    pad_9_mode_fail()
    pad_9_mode_nopad()
    pad_11_float()
    pad_11_int32()
    pad_11_int64()
    pad_11_double()
    pad_11_mode_edge()
