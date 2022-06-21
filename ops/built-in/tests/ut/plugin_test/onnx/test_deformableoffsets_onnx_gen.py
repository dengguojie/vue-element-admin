#deform_conv.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def make_deform_offsets_1():
    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'DeformableOffsets',
        inputs=['input', 'offset'],
        outputs=['Y'],
        deformable_groups=8,
        ksize=[3, 3],
        dilations=[1,1,1,1],
        im2col_step=32,
        pads=[1,1,1,1],
        strides=[1,1,1,1],
        name='test_deform_offsets_1'
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        nodes=[node_def],
        name='test_deform_offsets_1',
        inputs=[helper.make_tensor_value_info('input', TensorProto.FLOAT16, [4, 16,64,64]),
                helper.make_tensor_value_info('offset', TensorProto.FLOAT16, [4, 216,64,64])],
        outputs=[helper.make_tensor_value_info('Y', TensorProto.FLOAT16, [4, 32,192,192])]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-deform_offsets_1')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_deform_offsets_1_case.onnx")


if __name__ == "__main__":
    make_deform_offsets_1()
