# Given a bool scalar input cond.
# return constant tensor x if cond is True, otherwise return constant tensor y.
import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto

then_out = onnx.helper.make_tensor_value_info('then_out', onnx.TensorProto.FLOAT, [5])
else_out = onnx.helper.make_tensor_value_info('else_out', onnx.TensorProto.FLOAT, [5])
cond = onnx.helper.make_tensor_value_info('cond', onnx.TensorProto.FLOAT, [])
res = onnx.helper.make_tensor_value_info('res', onnx.TensorProto.FLOAT, [5])

x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

then_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['then_out'],
    value=onnx.numpy_helper.from_array(x)
)

else_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['else_out'],
    value=onnx.numpy_helper.from_array(y)
)

then_body = onnx.helper.make_graph(
    [then_const_node],
    'then_body',
    [],
    [then_out]
)

else_body = onnx.helper.make_graph(
    [else_const_node],
    'else_body',
    [],
    [else_out]
)

if_node = onnx.helper.make_node(
    'If',
    inputs=['cond'],
    outputs=['res'],
    then_branch=then_body,
    else_branch=else_body
)

graph_def = helper.make_graph(
        [if_node],
        'test_if',
        [cond],
        [res],
)
model_def = helper.make_model(graph_def, producer_name='if-onnx')
model_def.opset_import[0].version = 11
onnx.save(model_def, "./If.onnx")
