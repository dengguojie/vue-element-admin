import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def do_not_keepdims(version_t):
    axes = [1]
    keepdims = 0

    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [3, 2])

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data'],
        outputs=['reduced'],
        axes=axes,
        keepdims=keepdims)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data],
        outputs=[reduced])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./onnx/test_ReduceSum_onnx_case1.onnx")


def keepdims(version_t):  # type: () -> None
    axes = [1]
    keepdims = 1
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [3, 1, 2])

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data'],
        outputs=['reduced'],
        axes=axes,
        keepdims=keepdims)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data],
        outputs=[reduced])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./onnx/test_ReduceSum_onnx_case2.onnx")


def default_axes_keepdims(version_t):
    keepdims = 1
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [1, 1, 1])

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data'],
        outputs=['reduced'],
        keepdims=keepdims)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data],
        outputs=[reduced])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./onnx/test_ReduceSum_onnx_case3.onnx")


def negative_axes_keepdims(version_t):
    axes = [-2]
    keepdims = 1
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [3, 1, 2])

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data'],
        outputs=['reduced'],
        axes=axes,
        keepdims=keepdims)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data],
        outputs=[reduced])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceSum-onnx')
    model_def.opset_import[0].version = version_t

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./onnx/test_ReduceSum_onnx_case4.onnx")

def reducesum_e_axes1():
    keepdims = 1
    noop_with_empty_axes = 1
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 1])
    axes = helper.make_tensor('axes', TensorProto.INT64, [1], [1])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [1, 1, 1])

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data', 'axes'],
        outputs=['reduced'],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [data],
        [reduced],
        [axes],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceSum-onnx')
    model_def.opset_import[0].version = 13

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_ReduceSum_eaxes1_v13.onnx")

def reducesum_e_axes0():
    keepdims = 1
    noop_with_empty_axes = 0
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 1])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [1, 1, 1])

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data'],
        outputs=['reduced'],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [data],
        [reduced],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceSum-onnx')
    model_def.opset_import[0].version = 13

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_ReduceSum_eaxes0_v13.onnx")


if __name__ == "__main__":
    reducesum_e_axes0()
    reducesum_e_axes1()
    version_t = 11
    do_not_keepdims(version_t)
    keepdims(version_t)
    default_axes_keepdims(version_t)
    negative_axes_keepdims(version_t)
