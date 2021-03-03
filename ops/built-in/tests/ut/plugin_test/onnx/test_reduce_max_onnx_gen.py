import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# The protobuf definition can be found here:
# https://github.com/blob/master/onnx.proto


def do_not_keepdims(version_t):
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [3, 2])

    node = onnx.helper.make_node(
        'ReduceMax',
        inputs=['data'],
        outputs=['reduced'],
        axes=[1],
        keepdims=0)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data],
        outputs=[reduced]
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceMax-onnx')
    model_def.opset_import[0].version = version_t  # version

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_reduce_max_case_1.onnx")


def keepdims(version_t):  # type: () -> None
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [3, 1, 2])

    node = onnx.helper.make_node(
        'ReduceMax',
        inputs=['data'],
        outputs=['reduced'],
        axes=[1],
        keepdims=1)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data],
        outputs=[reduced])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceMax-onnx')
    model_def.opset_import[0].version = version_t  # version

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_reduce_max_case_2.onnx")


def default_axes_keepdims(version_t):
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [1, 1, 1])

    node = onnx.helper.make_node(
        'ReduceMax',
        inputs=['data'],
        outputs=['reduced'],
        # axes=[0, 1, 2],
        keepdims=1)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data],
        outputs=[reduced])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceMax-onnx')
    model_def.opset_import[0].version = version_t  # version

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_reduce_max_case_3.onnx")


def negative_axes_keepdims(version_t):
    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

    # Create one output (ValueInfoProto)
    reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [3, 1, 2])

    node = onnx.helper.make_node(
        'ReduceMax',
        inputs=['data'],
        outputs=['reduced'],
        axes=[-2],
        keepdims=1)

    graph_def = helper.make_graph(
        [node],
        'test-model',
        inputs=[data],
        outputs=[reduced])

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zl-ReduceMax-onnx')
    model_def.opset_import[0].version = version_t  # version

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_reduce_max_case_4.onnx")


if __name__ == "__main__":
    version_t = 12
    do_not_keepdims(version_t)
    keepdims(version_t)
    default_axes_keepdims(version_t)
    negative_axes_keepdims(version_t)
