#deform_conv.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_defrom_conv_1():
    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'DeformableConv2D',
        inputs=['input', 'weight', 'offset'],
        outputs=['Y'],
        deformable_groups=2,
        dilations=[3, 5],
        im2col_step=32,
        pads=[1, 1],
        strides=[2, 4],
        groups=32
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test_defrom_conv_1',
        inputs=[helper.make_tensor_value_info('input', TensorProto.FLOAT16, [1, 256,208,304]),
                helper.make_tensor_value_info('weight', TensorProto.FLOAT16, [256,8,3,5]),
                helper.make_tensor_value_info('offset', TensorProto.FLOAT16, [1, 54,102,74])],
        outputs=[helper.make_tensor_value_info('Y', TensorProto.FLOAT16, [1, 256,102,74])]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-defrom_conv_1')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_deform_conv_1_case.onnx")

if __name__ == "__main__":
    make_defrom_conv_1()
