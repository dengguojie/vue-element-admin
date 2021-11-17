# deform_conv.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_boundingbox_decode_1():
    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'BoundingBoxDecode',
        inputs=['input1', 'input2'],
        outputs=['Y'],
        max_shape=[1216, 1216],
        means=[0.0, 0.0, 0.0, 0.0],
        stds=[1.0, 1.0, 1.0, 1.0],
        wh_ratio_clip=0.016
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test_defrom_conv_1',
        inputs=[helper.make_tensor_value_info('input1', TensorProto.FLOAT, [5000, 4]),
                helper.make_tensor_value_info('input2', TensorProto.FLOAT, [5000, 4])],
        outputs=[helper.make_tensor_value_info(
            'Y', TensorProto.FLOAT, [5000, 4])]
    )
    # Create the model (ModelProto)
    model_def = helper.make_model(
        graph_def, producer_name='onnx-boundingbox_decode_1')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_boundingbox_decode_1.onnx")

if __name__ == "__main__":
    make_boundingbox_decode_1()