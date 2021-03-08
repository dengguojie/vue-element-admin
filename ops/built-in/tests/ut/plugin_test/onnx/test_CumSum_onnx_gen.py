import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def export_cumsum_1d(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, ])
    axis = helper.make_tensor('axis', TensorProto.INT32, [1, ], [0])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, ])

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'])

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [axis])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_CumSum_onnx_case1.onnx")


def cumsum_1d_exclusive(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, ])
    axis = helper.make_tensor('axis', TensorProto.INT32, [1, ], [0])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, ])

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
        exclusive=1)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [axis])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_CumSum_onnx_case2.onnx")


def cumsum_1d_reverse(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, ])
    axis = helper.make_tensor('axis', TensorProto.INT32, [1, ], [0])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, ])

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
        reverse=1)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [axis])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_CumSum_onnx_case3.onnx")


def cumsum_1d_reverse_exclusive(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, ])
    axis = helper.make_tensor('axis', TensorProto.INT32, [1, ], [0])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, ])

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
        reverse=1)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [axis])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_CumSum_onnx_case4.onnx")


def cumsum_2d_axis_0(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    axis = helper.make_tensor('axis', TensorProto.INT32, [1, ], [0])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [axis])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_CumSum_onnx_case5.onnx")


def cumsum_2d_axis_1(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    axis = helper.make_tensor('axis', TensorProto.INT32, [1, ], [1])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [axis])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_CumSum_onnx_case6.onnx")


def cumsum_2d_negative_axis(version_t):
    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    axis = helper.make_tensor('axis', TensorProto.INT32, [1, ], [-1])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'])

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
        [axis])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-CumSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_CumSum_onnx_case7.onnx")


if __name__ == "__main__":
    version = 12
    export_cumsum_1d(version)
    cumsum_1d_exclusive(version)
    cumsum_1d_reverse(version)
    cumsum_1d_reverse_exclusive(version)
    cumsum_2d_axis_0(version)
    cumsum_2d_axis_1(version)
    cumsum_2d_negative_axis(version)
